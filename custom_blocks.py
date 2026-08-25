import sys
import numpy as np
import adi
from gnuradio import gr, blocks

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

        self.samp_rate = samp_rate
        self.freq_offset_hz = freq_offset_hz
        self.phase = 0.0
        self.phase_inc = 2 * np.pi * freq_offset_hz / samp_rate

    def work(self, input_items, output_items):
        tx = input_items[0]
        rx = output_items[0]

        n = len(tx)
        out_n = len(rx)

        # --- 向量化寫入 FIFO ---
        w_end = self.write_idx + n
        if w_end < self.fifo_size:
            self.buffer[self.write_idx:w_end] = tx
        else:
            first = self.fifo_size - self.write_idx
            self.buffer[self.write_idx:] = tx[:first]
            self.buffer[:n-first] = tx[first:]
        self.write_idx = (self.write_idx + n) % self.fifo_size

        # --- 頻偏向量化 ---
        t = np.arange(out_n)
        phases = self.phase + self.phase_inc * t
        self.phase = (phases[-1] + self.phase_inc) % (2 * np.pi)
        rot = np.exp(1j * phases)

        # --- 向量化讀出 FIFO ---
        r_end = self.read_idx + out_n
        if r_end < self.fifo_size:
            rx[:] = self.buffer[self.read_idx:r_end] * rot
        else:
            first = self.fifo_size - self.read_idx
            rx[:first] = self.buffer[self.read_idx:] * rot[:first]
            rx[first:] = self.buffer[:out_n-first] * rot[first:]
        self.read_idx = (self.read_idx + out_n) % self.fifo_size

        return out_n
