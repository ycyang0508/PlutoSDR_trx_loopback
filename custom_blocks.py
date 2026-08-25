import sys
import numpy as np
import adi
import sip
from gnuradio import gr, blocks

# ---------------------------------------------------------
# Fake PlutoSDR with FIFO + frequency offset
# ---------------------------------------------------------
class FakePlutoSDR(gr.sync_block):
    def __init__(self, fifo_size=65536, samp_rate=1_000_000, freq_offset_hz=2000.0):
        gr.sync_block.__init__(
            self,
            name="FakePlutoSDR_FIFO_FreqOffset",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )

        self.fifo_size = fifo_size
        self.buffer = np.zeros(fifo_size, dtype=np.complex64)
        self.write_idx = 0
        self.read_idx = 0

        # frequency offset
        self.samp_rate = samp_rate
        self.freq_offset_hz = freq_offset_hz
        self.phase = 0.0
        self.phase_inc = 2 * np.pi * freq_offset_hz / samp_rate

    def work(self, input_items, output_items):
        tx = input_items[0]
        rx = output_items[0]

        n = len(tx)
        out_n = len(rx)

        # write TX → FIFO
        for i in range(n):
            self.buffer[self.write_idx] = tx[i]
            self.write_idx = (self.write_idx + 1) % self.fifo_size

        # read FIFO → RX + freq offset
        for i in range(out_n):
            val = self.buffer[self.read_idx]
            self.read_idx = (self.read_idx + 1) % self.fifo_size

            rx[i] = val * np.exp(1j * self.phase)
            self.phase += self.phase_inc
            if self.phase > 2 * np.pi:
                self.phase -= 2 * np.pi

        return out_n




class PlutoSDR_txrx_stream(gr.hier_block2):
    def __init__(self, uri="ip:192.168.2.1", samp_rate=1_000_000,
                 tx_lo=915e6, rx_lo=915e6, buf_len=16384):

        gr.hier_block2.__init__(
            self,
            "pluto_txrx_stream",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
        )

        self.buf_len = buf_len

        # Pluto SDR 初始化
        self.sdr = adi.Pluto(uri)
        self.sdr.sample_rate = int(samp_rate)

        # TX 設定
        self.sdr.tx_lo = int(tx_lo)
        self.sdr.tx_hardwaregain_chan0 = -10
        self.sdr.tx_buffer_size = buf_len
        self.sdr.tx_cyclic_buffer = False

        # RX 設定
        self.sdr.rx_lo = int(rx_lo)
        self.sdr.gain_control_mode_chan0 = 'manual'
        self.sdr.rx_hardwaregain_chan0 = 10
        self.sdr.rx_buffer_size = buf_len
        self.sdr.rx_rf_bandwidth = int(samp_rate * 2)

        # GNU Radio 內建乘法 block（TX 放大）
        self.tx_scale = blocks.multiply_const_cc(2**14)

        # GNU Radio 內建乘法 block（RX 縮小）
        self.rx_scale = blocks.multiply_const_cc(1/(2**14))

        # Pluto I/O block（純 Python）
        self.pluto_io = PlutoIO(buf_len, self.sdr)

        # Connect internal blocks
        self.connect(self, self.tx_scale, self.pluto_io, self.rx_scale, self)
        

class PlutoIO(gr.sync_block):
    def __init__(self, buf_len, sdr):
        gr.sync_block.__init__(
            self,
            name="pluto_io",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self.buf_len = buf_len
        self.sdr = sdr
        self.set_output_multiple(buf_len)

    def work(self, input_items, output_items):
        in_data = input_items[0]
        out_data = output_items[0]
        n_out = len(out_data)

        for i in range(0, n_out, self.buf_len):
            end = min(i + self.buf_len, n_out)

            tx_chunk = in_data[i:end]
            if len(tx_chunk) > 0:
                self.sdr.tx(tx_chunk)

            rx_chunk = self.sdr.rx()
            out_data[i:end] = rx_chunk[:(end - i)]

        return n_out
