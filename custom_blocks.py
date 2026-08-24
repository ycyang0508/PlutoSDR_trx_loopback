import sys
import numpy as np
import adi
import sip
from gnuradio import gr



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