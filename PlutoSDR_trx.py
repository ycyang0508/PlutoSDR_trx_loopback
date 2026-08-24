import sys
import numpy as np
import adi
import sip
from gnuradio import gr


class PlutoSDR_txrx(gr.sync_block):
    def __init__(self, uri="ip:192.168.2.1", samp_rate=1e6,
                 tx_lo=2.4e9, rx_lo=2.4e9, buf_len=16384):

        gr.sync_block.__init__(
            self,
            name="pluto_txrx",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )

        # GNU Radio buffer multiple
        self.buf_len = buf_len
        self.set_output_multiple(buf_len)

        # Pluto SDR 初始化
        self.sdr = adi.Pluto(uri)
        self.sdr.sample_rate = int(samp_rate)
        self.sdr.loopback = 0        

        # TX setup
        self.sdr.tx_lo = int(tx_lo)
        self.sdr.tx_hardwaregain_chan0 = -30
        self.sdr.tx_cyclic_buffer = False
        self.sdr.tx_buffer_size = buf_len
        self.sdr.tx_destroy_buffer()

        # RX setup
        self.sdr.rx_lo = int(rx_lo)
        self.sdr.gain_control_mode_chan0 = 'manual'
        #self.sdr.gain_control_mode_chan0 = "fast_attack"
        self.sdr.rx_hardwaregain_chan0 = 54
        self.sdr.rx_buffer_size = buf_len
        self.sdr.rx_rf_bandwidth = 6_000_000
        self.sdr.rx_destroy_buffer()


    def work(self, input_items, output_items):
        in_data = input_items[0]
        out_data = output_items[0]

        n_out = len(out_data)
        n_in = len(in_data)
        chunk = self.buf_len

        # 若 input 比 output 小 → 自動補零
        if n_in < n_out:
            in_data = np.pad(in_data, (0, n_out - n_in), mode='constant')

        # 分 chunk 處理
        for i in range(0, n_out, chunk):

            tx_chunk = in_data[i:i+chunk]

            # 對齊 chunk 長度
            if len(tx_chunk) < chunk:
                tx_chunk = np.pad(tx_chunk, (0, chunk - len(tx_chunk)), mode='constant')

            # TX
            self.sdr.tx(tx_chunk)

            # RX
            rx_chunk = self.sdr.rx()

            # 填入 output buffer
            end = min(i + chunk, n_out)
            out_data[i:end] = rx_chunk[:(end - i)]

        return n_out

