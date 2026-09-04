import time
import threading
import numpy as np
from gnuradio import gr, blocks, analog
import pmt


# ==============================================================================
# 1. Custom Block 實作：1-to-N Dynamic Demux
# ==============================================================================
class dynamic_demux(gr.basic_block):
    def __init__(self, num_outputs=3):
        gr.basic_block.__init__(
            self,
            name="Dynamic 1-to-N Demux",
            in_sig=[np.complex64],                  # 1 個輸入埠
            out_sig=[np.complex64] * num_outputs   # N 個輸出埠
        )
        self.num_outputs = num_outputs
        self.active_path = 0  # 預設輸出的路徑 index (0 ~ N-1)

        # 註冊 Message Port 用於接收控制指令
        self.message_port_register_in(pmt.intern("cmd"))
        self.set_msg_handler(pmt.intern("cmd"), self.handle_cmd)

    def handle_cmd(self, msg):
        """處理切換指令"""
        if pmt.is_integer(msg):
            new_path = pmt.to_long(msg)
            if 0 <= new_path < self.num_outputs:
                self.active_path = new_path
                print(f"\n[DEMUX] ➔ 切換至 Output Port [{self.active_path}]")
            else:
                print(f"\n[DEMUX Warning] 無效的路徑 Index: {new_path}")

    def general_work(self, input_items, output_items):
        in0 = input_items[0]
        ninput_items = len(in0)
        if ninput_items == 0:
            return 0

        # 計算當前被選中的 output port 最多能接收多少 samples
        noutput_items = min(ninput_items, len(output_items[self.active_path]))
        if noutput_items == 0:
            return 0

        # 將資料複製到被選中的 Port
        output_items[self.active_path][:noutput_items] = in0[:noutput_items]

        # 計算各 Port 產生的數據量 (未選中的 Port 為 0)
        produced_counts = [0] * self.num_outputs
        produced_counts[self.active_path] = noutput_items

        # 消耗輸入端的 samples
        self.consume(0, noutput_items)

        # 告知 GNU Radio 各埠產生的量
        for port_idx, count in enumerate(produced_counts):
            self.produce(port_idx, count)

        return gr.WORK_CALLED_PRODUCE


# ==============================================================================
# 2. Command 發送器：輪流切換 Port 0 -> Port 1 -> Port 2
# ==============================================================================
class path_switcher(gr.basic_block):
    """自訂控制 Block，使用背景 Thread 定時發送切換 Message"""
    def __init__(self, num_paths=3, interval_sec=2.0):
        gr.basic_block.__init__(self, name="Path Switcher", in_sig=None, out_sig=None)
        self.num_paths = num_paths
        self.interval = interval_sec
        self.current_path = 0
        self.running = True

        self.message_port_register_out(pmt.intern("cmd_out"))

        # 啟動背景線程獨立定時發送 Message
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def _run(self):
        while self.running:
            time.sleep(self.interval)
            # 發送切換 Command
            msg = pmt.from_long(self.current_path)
            self.message_port_pub(pmt.intern("cmd_out"), msg)

            # 遞增 Port Index
            self.current_path = (self.current_path + 1) % self.num_paths

    def stop(self):
        self.running = False
        return super().stop()

# ==============================================================================
# 3. 測試環境搭建 (Flowgraph)
# ==============================================================================
class demux_test_flowgraph(gr.top_block):
    def __init__(self):
        super().__init__("Demux Test Flowgraph")

        sample_rate = 32000
        num_ports = 3

        # 1. 訊號源與 Throttle
        self.src = analog.sig_source_c(sample_rate, analog.GR_COS_WAVE, 1000, 1.0)
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, sample_rate, True)

        # 2. Demux 與 Switcher
        self.demux = dynamic_demux(num_outputs=num_ports)
        self.switcher = path_switcher(num_paths=num_ports, interval_sec=2.0)

        # 3. 連接 Message Port 與主資料流
        self.connect(self.src, self.throttle, self.demux)
        self.msg_connect(self.switcher, "cmd_out", self.demux, "cmd")

        # 4. 連接各 Output Port
        self.probes = []
        for i in range(num_ports):
            probe = blocks.probe_rate(gr.sizeof_gr_complex, 500.0)
            self.connect((self.demux, i), probe)
            self.probes.append(probe)

    def print_status(self):
        """印出當前各 Output Port 的數據傳輸速率 (samples/sec)"""
        rates = [int(p.rate()) for p in self.probes]
        print(f"Port 數據速率 (samples/s) ➔ Port 0: {rates[0]:<6} | Port 1: {rates[1]:<6} | Port 2: {rates[2]:<6}", end="\r")


# ==============================================================================
# 4. 執行測試
# ==============================================================================
if __name__ == "__main__":
    tb = demux_test_flowgraph()
    print("=== 開始執行 Demux 測試環境 (每 2 秒切換一次 Port) ===")
    tb.start()

    try:
        for _ in range(30): # 測試執行 15 秒
            tb.print_status()
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    tb.stop()
    tb.wait()
    print("\n=== 測試結束 ===")