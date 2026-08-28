#!/usr/bin/env python3
import sys
import numpy as np
import adi
from PyQt5 import Qt
import sip
import time
from gnuradio import gr, blocks, analog, digital, qtgui, filter, fft,zeromq
from custom_blocks import *
from collections import deque
from radio_eval import *
from rf_trx import *
from qpsk_symbol_dsp import *
from QAM16_symbol_dsp import *
from QAM64_symbol_dsp import *
from QAM256_symbol_dsp import *
# ---------------------------------------------------------
#  主程式
# ---------------------------------------------------------
class qpsk_cable_demo(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self)

        samp_rate = 1_000_000
        sps = 4
        buf_len = 16384
              
        #self.sym_dsp_tx = QPSK_TX_block(sps=sps, samp_rate=samp_rate)
        #self.sym_dsp_rx = QPSK_RX_block(sps=sps, samp_rate=samp_rate)
        #self.sym_dsp_tx = QAM16_TX_block(sps=sps, samp_rate=samp_rate)
        #self.sym_dsp_rx = QAM16_RX_block(sps=sps, samp_rate=samp_rate)
        #self.sym_dsp_tx = QAM64_TX_block(sps=sps, samp_rate=samp_rate)
        #self.sym_dsp_rx = QAM64_RX_block(sps=sps, samp_rate=samp_rate)
        self.sym_dsp_tx = QAM256_TX_block(sps=sps, samp_rate=samp_rate)
        self.sym_dsp_rx = QAM256_RX_block(sps=sps, samp_rate=samp_rate)

        n_symbols = 200000
        constellation_point = self.sym_dsp_rx.constellation_point
        rnd = np.random.randint(0, constellation_point, n_symbols).tolist()
        self.src = blocks.vector_source_b(rnd, repeat=True)

        self.pluto = PlutoSDR_txrx_stream(
            uri="ip:192.168.1.10",
            samp_rate=samp_rate,
            tx_lo=915e6,
            rx_lo=915e6,
            buf_len=buf_len
        )        
        self.freq_sink = qtgui.freq_sink_c(
            8192,
            fft.window.WIN_HAMMING,
            0,
            samp_rate,
            "Rx Spectrum (1MHz)"
        )
        self.freq_win = sip.wrapinstance(self.freq_sink.qwidget(), Qt.QWidget)
        self.freq_sink.set_fft_average(0.3)

        self.const_sink = qtgui.const_sink_c(
            1024,
            "Cable Loopback 256QAM (1MHz)",
            1
        )
        self.const_sink.set_x_axis(-2.0, 2.0)
        self.const_sink.set_y_axis(-2.0, 2.0)
        self.const_win = sip.wrapinstance(self.const_sink.qwidget(), Qt.QWidget)

        const = self.sym_dsp_rx.const
        raw_pts = np.array(const.points(), dtype=np.complex64)
        points = raw_pts / np.abs(raw_pts[0])
        self.evm = evm_generic_block(points, window=2048, skip_samples=32768)

        self.evm_sink = qtgui.number_sink(
            gr.sizeof_float,
            0.5,
            qtgui.NUM_GRAPH_HORIZ,
            1,
            None
        )
        self.evm_sink.set_title("EVM (%)")
        self.evm_sink.set_update_time(2.0)
        self.evm_sink.set_min(0, -20)
        self.evm_sink.set_max(0, 20)
        self.evm_sink_win = sip.wrapinstance(self.evm_sink.qwidget(), Qt.QWidget)
               

        # 新增：bit output 的空 sink
        self.bit_sink = blocks.null_sink(gr.sizeof_char)

        # TX path
        self.connect(self.src, self.sym_dsp_tx, self.pluto)

        # Spectrum
        self.connect(self.pluto, self.freq_sink)

        # RX symbols + bits
        self.connect(self.pluto, self.sym_dsp_rx)        

        # constellation
        self.connect((self.sym_dsp_rx, 0), (self.const_sink, 0))

        # EVM
        self.connect((self.sym_dsp_rx, 0), self.evm,self.evm_sink)        

        # 新增：bit output 接到 null sink
        self.connect((self.sym_dsp_rx, 1), self.bit_sink)

def run_demo():
    app = Qt.QApplication(sys.argv)
    tb = qpsk_cable_demo()

    win = Qt.QWidget()
    layout = Qt.QVBoxLayout(win)
    layout.addWidget(tb.freq_win)
    layout.addWidget(tb.const_win)
    layout.addWidget(tb.evm_sink_win)
    win.show()

    tb.start()
    app.exec_()
    tb.stop()
    tb.wait()

class zeroMQ_direct_loopback(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self)

        uri="ip:192.168.1.10"  # 保留介面相容性（內部不使用）
        samp_rate=1_000_000
        tx_lo=915e6
        rx_lo=915e6
        buf_len=32768
        zmq_rx_addr="tcp://127.0.0.1:5555"
        zmq_tx_addr="tcp://127.0.0.1:5556"

        # Pluto TX/RX
        #self.pluto = PlutoSDR_zmq_txrx_stream()

        # 1. 實例化內部 Buffer Loopback 核心
        self.loopback_core = InternalBufferLoopback(buf_len=buf_len)

        # 2. Rate Control (Throttle 確保讀寫速度符合 sample_rate)
        self.throttle = blocks.throttle(
            gr.sizeof_gr_complex, samp_rate, True
        )

        # 3. ZeroMQ 端點配置
        # TX: SUB Source 接收 Python PUB (bind=False)
        self.zmq_tx_source = zeromq.pull_source(gr.sizeof_gr_complex,  1, zmq_tx_addr, 2048, False, -1, False )

        # RX: PUSH Sink 傳送給 Python PULL (bind=True)
        self.zmq_rx_sink = zeromq.push_sink(
            gr.sizeof_gr_complex, 1, zmq_rx_addr, 2048, False, -1, True
        )

        # 4. 拓撲連線：ZMQ TX Source -> Loopback Buffer -> Throttle -> ZMQ RX Sink
        self.connect(self.zmq_tx_source, self.loopback_core)
        self.connect(self.loopback_core, self.throttle, self.zmq_rx_sink)

class zeroMQ_PlutoSDR_demo(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self, "zeroMQ_PlutoSDR_demo")

        uri = "ip:192.168.1.10"
        samp_rate = 1_000_000
        tx_lo = 915e6
        rx_lo = 915e6
        buf_len = 32768
        zmq_rx_addr = "tcp://127.0.0.1:5555"
        zmq_tx_addr = "tcp://127.0.0.1:5556"

        # 1. 實例化純粹處理硬體 I/O 的 Pluto Stream Block
        self.pluto = PlutoSDR_txrx_stream(
            uri=uri,
            samp_rate=samp_rate,
            tx_lo=tx_lo,
            rx_lo=rx_lo,
            buf_len=buf_len,
        )

        # 2. Rate Control (Throttle 確保讀寫速度與 Sample Rate 對齊)
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)

        # 3. ZeroMQ 端點配置 (直接置於 Top Block)
        # TX: PULL Source 接收來自 Python PUSH 的資料 (bind=False)
        self.zmq_tx_source = zeromq.pull_source(
            gr.sizeof_gr_complex, 1, zmq_tx_addr, buf_len, False, -1, False
        )

        # RX: PUSH Sink 傳送資料給 Python PULL (bind=True)
        self.zmq_rx_sink = zeromq.push_sink(
            gr.sizeof_gr_complex, 1, zmq_rx_addr, buf_len, False, -1, True
        )

        # 4. 拓撲連線：ZMQ TX Source -> Pluto TX | Pluto RX -> Throttle -> ZMQ RX Sink
        # 註：若 PlutoSDR_txrx_stream 內部有分別處理 TX/RX 的 pad/in/out，依對應端口連接：
        self.connect(self.zmq_tx_source, self.pluto)
        self.connect(self.pluto, self.throttle, self.zmq_rx_sink)

def run_zeroMQ():
    tb = zeroMQ_PlutoSDR_demo()
    tb.start()
    try:
        print("Flowgraph running... Press Ctrl-C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping flowgraph...")
        tb.stop()
        tb.wait()

if __name__ == "__main__":
    run_demo()
    #run_zeroMQ()
