import sys
import numpy as np
import adi
import sip
from gnuradio import gr, blocks

class q15_tx(gr.sync_block):
    """
    TX: complex64 → Q1.15 int16
    """


    def __init__(self):
        gr.sync_block.__init__(
            self,
            name="q15_tx",
            in_sig=[np.complex64],
            out_sig=[np.complex64]   # Pluto TX block 仍然吃 complex64，但內容是 int16
        )
        self.scale = 2**15 #32768.0 


    def work(self, input_items, output_items):
        x = input_items[0]

        # float → Q1.15
        i_float = np.clip(x.real, -1.0, 0.999969) * self.scale
        q_float = np.clip(x.imag, -1.0, 0.999969) * self.scale

        i_int16 = i_float.astype(np.int16)
        q_int16 = q_float.astype(np.int16)

        # 轉成 complex64，但內容是 int16
        output_items[0][:] = i_int16.astype(np.float32) + 1j * q_int16.astype(np.float32)

        return len(output_items[0])


class q15_rx(gr.sync_block):
    """
    RX: Q1.15 int16 → complex64
    """

    def __init__(self):
        gr.sync_block.__init__(
            self,
            name="q15_rx",
            in_sig=[np.complex64],   # Pluto RX block 輸出 complex64，但內容是 int16
            out_sig=[np.complex64]
        )
        self.scale = 2**5  #32767.0

    def work(self, input_items, output_items):
        x = input_items[0]

        # complex64 → int16
        i_int16 = x.real.astype(np.int16)
        q_int16 = x.imag.astype(np.int16)

        # int16 → float
        i_float = i_int16.astype(np.float32) / self.scale
        q_float = q_int16.astype(np.float32) / self.scale

        output_items[0][:] = i_float + 1j * q_float
        return len(output_items[0])



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