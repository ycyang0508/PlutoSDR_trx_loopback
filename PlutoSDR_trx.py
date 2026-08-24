import sys
import numpy as np
import adi
import sip
from gnuradio import gr

class PlutoSDR_txrx_stream(gr.sync_block):
    def __init__(self, uri="ip:192.168.2.1", samp_rate=1_000_000,
                 tx_lo=915e6, rx_lo=915e6, buf_len=16384):

        # 設有 1 個 Input (np.complex64) 與 1 個 Output (np.complex64)
        gr.sync_block.__init__(
            self,
            name="pluto_txrx_stream",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )

        self.buf_len = buf_len
        self.set_output_multiple(buf_len)

        # 1. Pluto SDR 初始化
        self.sdr = adi.Pluto(uri)
        self.sdr.sample_rate = int(samp_rate)
        
        # 2. TX 硬體設定
        self.sdr.tx_lo = int(tx_lo)
        self.sdr.tx_hardwaregain_chan0 = -10   
        self.sdr.tx_buffer_size = buf_len
        self.sdr.tx_cyclic_buffer = False   # 關閉硬體循環，採用動態串流發射

        # 3. RX 硬體設定
        self.sdr.rx_lo = int(rx_lo)
        self.sdr.gain_control_mode_chan0 = 'manual'
        self.sdr.rx_hardwaregain_chan0 = 10    
        self.sdr.rx_buffer_size = buf_len
        self.sdr.rx_rf_bandwidth = int(samp_rate * 2)

    def work(self, input_items, output_items):
        in_data = input_items[0]   # 接收來自 GNU Radio 上游 (tx_src) 的 TX 資料
        out_data = output_items[0]
        n_out = len(out_data)

        for i in range(0, n_out, self.buf_len):
            end = min(i + self.buf_len, n_out)
            
            # A. 抓取上游傳入的 TX 訊號，放大後推送至 PlutoSDR 進行即時發射
            tx_chunk = in_data[i:end]
            if len(tx_chunk) > 0:
                self.sdr.tx(tx_chunk * 2**14)

            # B. 同步讀取 RX 資料並歸一化輸出給下游 DSP
            rx_chunk = self.sdr.rx()
            out_data[i:end] = rx_chunk[:(end - i)] / 10000.0

        return n_out