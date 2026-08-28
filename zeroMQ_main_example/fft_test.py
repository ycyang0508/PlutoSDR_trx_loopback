import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
import zmq

# --- 全域參數 ---
samp_rate = 1_000_000
buf_len = 32768  # 發送與期望的 FFT 點數長度
tone_freq = 100000  # 100 kHz TX 正弦波


# --- 背景 ZMQ 收發執行緒 (含類 FIFO 接收機制) ---
class ZMQWorkerThread(QThread):
    data_received = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.is_running = True
        # 初始化類 FIFO 緩衝區 (預設為 complex64 空陣列)
        self.rx_fifo = np.array([], dtype=np.complex64)

    def run(self):
        context = zmq.Context()

        # 1. TX: PUSH Mode
        socket_tx = context.socket(zmq.PUSH)
        socket_tx.bind("tcp://127.0.0.1:5556")

        # 2. RX: PULL Mode
        socket_rx = context.socket(zmq.PULL)
        socket_rx.setsockopt(zmq.RCVBUF, 65536 * 4)  # 放大底層 Socket Buffer
        socket_rx.connect("tcp://127.0.0.1:5555")

        # 產生 TX 發送訊號
        t = np.arange(buf_len)
        tx_signal = (0.8 * np.exp(1j * 2 * np.pi * tone_freq * t / samp_rate)).astype(
            np.complex64
        )
        tx_bytes = tx_signal.tobytes()

        while self.is_running:
            # --- A. 發送 TX ---
            try:
                socket_tx.send(tx_bytes, flags=zmq.NOBLOCK)
            except (zmq.Again, zmq.ZMQError):
                pass

            # --- B. 接收 RX 並推入 FIFO ---
            try:
                rx_bytes = socket_rx.recv(flags=zmq.NOBLOCK)
                if rx_bytes:
                    rx_chunk = np.frombuffer(rx_bytes, dtype=np.complex64)
                    # 1. Push: 將新收到的 chunk 拼接至 FIFO 尾端
                    self.rx_fifo = np.concatenate([self.rx_fifo, rx_chunk])
            except (zmq.Again, zmq.ZMQError):
                pass

            # --- C. 從 FIFO 吐出固定點數 (Pop) 給 GUI ---
            # 只要 FIFO 內的總點數滿 1024 點（或要畫圖的目標點數，如 buf_len）
            target_pts = 4096  # 可依據 FFT 解析度需求彈性調整 (例如 1024, 4096, 32768)
            if len(self.rx_fifo) >= target_pts:
                # 2. Extract: 擷取頭部的 target_pts 點
                out_chunk = self.rx_fifo[:target_pts]
                # 3. Pop: 移除已使用的數據，保留剩餘資料
                self.rx_fifo = self.rx_fifo[target_pts:]

                # 拋出給 UI 畫圖
                self.data_received.emit(out_chunk)

            # 防止 FIFO 溢位 (Overrun)：若累積超過 10 倍 target_pts，強制拋棄舊資料保證即時性
            if len(self.rx_fifo) > target_pts * 10:
                self.rx_fifo = self.rx_fifo[-target_pts:]

            self.msleep(10)  # 約 100 FPS 的輪詢頻率

        socket_tx.close()
        socket_rx.close()
        context.term()

    def stop(self):
        self.is_running = False
        self.wait()


# --- PyQtGraph 主繪圖視窗 ---
class SDRMainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PlutoSDR Real-Time ZMQ Spectrum (FIFO Mode)")
        self.resize(900, 550)

        layout = QVBoxLayout(self)
        self.graphWidget = pg.PlotWidget()
        layout.addWidget(self.graphWidget)

        self.graphWidget.setLabel("bottom", "Frequency", units="Hz")
        self.graphWidget.setLabel("left", "Magnitude", units="dB")
        self.graphWidget.setYRange(-80, 10)
        self.graphWidget.setXRange(-samp_rate / 2, samp_rate / 2)
        self.graphWidget.showGrid(x=True, y=True)

        self.curve = self.graphWidget.plot(pen="y")

        # 啟動背景 ZMQ 通訊
        self.worker = ZMQWorkerThread()
        self.worker.data_received.connect(self.update_plot)
        self.worker.start()

    def update_plot(self, rx_buffer):
        N = len(rx_buffer)

        freqs = np.fft.fftshift(np.fft.fftfreq(N, 1 / samp_rate))
        fft_raw = np.fft.fftshift(np.fft.fft(rx_buffer))
        mag = np.abs(fft_raw)
        max_val = np.max(mag) if np.max(mag) > 0 else 1.0
        fft_dB = 20 * np.log10((mag / max_val) + 1e-12)

        self.curve.setData(freqs, fft_dB)

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SDRMainWindow()
    win.show()
    sys.exit(app.exec_())