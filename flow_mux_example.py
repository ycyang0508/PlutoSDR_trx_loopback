
import time
import threading
import numpy as np
from gnuradio import gr, blocks, analog
import pmt

# ==============================================================================
# 1. Custom Block：無損 N-to-1 Dynamic Mux (不 consume 未選中 Port)
# ==============================================================================
class no_drop_mux(gr.basic_block):
    def __init__(self, num_inputs=3):
        gr.basic_block.__init__(
            self,
            name="No-Drop Dynamic Mux",
            in_sig=[np.complex64] * num_inputs,   # N 個輸入埠
            out_sig=[np.complex64]                # 1 個輸出埠
        )
        self.num_inputs = num_inputs
        self.active_path = 0  # 預設選中的輸入路徑 index (0 ~ N-1)

        # 註冊 Message Port 用於接收控制指令
        self.message_port_register_in(pmt.intern("cmd"))
        self.set_msg_handler(pmt.intern("cmd"), self.handle_cmd)

    def handle_cmd(self, msg):
        """處理切換指令"""
        if pmt.is_integer(msg):
            new_path = pmt.to_long(msg)
            if 0 <= new_path < self.num_inputs:
                self.active_path = new_path
                print(f"\n[MUX] ➔ 切換至 Input Port [{self.active_path}]")
            else:
                print(f"\n[MUX Warning] 無效的路徑 Index: {new_path}")

    def general_work(self, input_items, output_items):
        out0 = output_items[0]
        in_active = input_items[self.active_path]

        # 1. 檢查目前被選中的 Port 是否有 Samples 可用
        ninput_items = len(in_active)
        if ninput_items == 0:
            return 0  # 沒資料直接 return，完全不動作

        # 2. 計算輸出 Buffer 最多能放多少 Samples
        noutput_items = min(ninput_items, len(out0))
        if noutput_items == 0:
            return 0

        # 3. 複製選中 Port 的資料到 Output
        out0[:noutput_items] = in_active[:noutput_items]

        # 4. **核心差異**：僅 consume 當前被選中的 Port！
        # 完全不呼叫未選中 Port 的 consume()，讓 Samples 留在 Buffer 累積直到觸發 Blocking
        self.consume(self.active_path, noutput_items)

        return noutput_items


# ==============================================================================
# 2. Command 發送器 (背景 Thread)
# ==============================================================================
class path_switcher(gr.basic_block):
    def __init__(self, num_paths=3, interval_sec=3.0):
        gr.basic_block.__init__(self, name="Path Switcher", in_sig=None, out_sig=None)
        self.num_paths = num_paths
        self.interval = interval_sec
        self.current_path = 0
        self.running = True

        self.message_port_register_out(pmt.intern("cmd_out"))

        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def _run(self):
        while self.running:
            time.sleep(self.interval)
            msg = pmt.from_long(self.current_path)
            self.message_port_pub(pmt.intern("cmd_out"), msg)
            self.current_path = (self.current_path + 1) % self.num_paths

    def stop(self):
        self.running = False
        return super().stop()


# ==============================================================================
# 3. 測試環境搭建 (設定 Buffer 上限)
# ==============================================================================
class mux_test_flowgraph(gr.top_block):
    def __init__(self):
        super().__init__("No-Drop Mux Test Flowgraph")

        sample_rate = 32000
        num_inputs = 3
        # 設定 Buffer 容量上限為 8192 個 complex samples (約可緩衝 0.25 秒的數據)
        max_buffer_size = 8192  

        # 1. 訊號源與 Throttle
        self.sources = [
            analog.sig_source_c(sample_rate, analog.GR_COS_WAVE, 1000 * (i + 1), 1.0)
            for i in range(num_inputs)
        ]
        self.throttles = [
            blocks.throttle(gr.sizeof_gr_complex, sample_rate, True)
            for _ in range(num_inputs)
        ]

        # 2. Mux 與 Switcher
        self.mux = no_drop_mux(num_inputs=num_inputs)
        self.switcher = path_switcher(num_paths=num_inputs, interval_sec=3.0)

        # 3. 連接訊號源並為每一個 Mux Input Port 限制 Max Buffer
        for i in range(num_inputs):
            self.connect(self.sources[i], self.throttles[i])
            self.connect((self.throttles[i], 0), (self.mux, i))
            
            # **關鍵點**：限制該輸入端連接的 Output Buffer 最大容量
            # 當未選中的 Port 緩衝區達到 max_buffer_size 後，Throttle 也會被 Block 停止推進            
            self.throttles[i].set_max_output_buffer(max_buffer_size)

        # 4. 連接 Message Port
        self.msg_connect(self.switcher, "cmd_out", self.mux, "cmd")

        # 5. Output -> Probe -> Null Sink
        self.probe = blocks.probe_rate(gr.sizeof_gr_complex, 500.0)
        #self.null_sink = blocks.null_sink(gr.sizeof_gr_complex)
        self.connect(self.mux, self.probe)

    def print_status(self):
        """印出當前 Mux 總輸出的速率 (samples/sec)"""
        rate = int(self.probe.rate())
        print(f"Mux 輸出數據速率 ➔ {rate:<6} samples/s", end="\r")


# ==============================================================================
# 4. 執行測試
# ==============================================================================
if __name__ == "__main__":
    tb = mux_test_flowgraph()
    print("=== 開始執行 No-Drop Mux 測試環境 (每 3 秒切換一次 Port，Buffer 上限 = 8192) ===")
    tb.start()

    try:
        for _ in range(30):  # 測試執行 15 秒
            tb.print_status()
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    tb.stop()
    tb.wait()
    print("\n=== 測試結束 ===")