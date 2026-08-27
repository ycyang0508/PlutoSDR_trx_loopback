import sys
import numpy as np
import adi
from PyQt5 import Qt
import sip
from gnuradio import gr, blocks, analog, digital, qtgui, filter, fft
from custom_blocks import *
from collections import deque
from radio_eval import *
from rf_trx import *


# ---------------------------------------------------------
#  QPSK TX Block
# ---------------------------------------------------------
class QPSK_TX_block(gr.hier_block2):
    def __init__(self, sps=4, samp_rate=1_000_000, rolloff=0.35):
        gr.hier_block2.__init__(
            self,
            "QPSK_TX_block",
            gr.io_signature(1, 1, gr.sizeof_char),
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
        )

        sym_rate = samp_rate // sps
        ntaps = 15 * sps + 1

        const = digital.constellation_qpsk().base()
        points = const.points()

        self.mapper = digital.chunks_to_symbols_bc(points, 1)

        self.rrc_tx = filter.interp_fir_filter_ccf(
            sps,
            filter.firdes.root_raised_cosine(
                gain=1.0,
                sampling_freq=samp_rate,
                symbol_rate=sym_rate,
                alpha=rolloff,
                ntaps=ntaps
            )
        )

        self.connect(self, self.mapper, self.rrc_tx, self)

# ---------------------------------------------------------
#  QPSK RX Block（輸出 symbols + bits）
# ---------------------------------------------------------
class QPSK_RX_block(gr.hier_block2):
    def __init__(self, sps=4, samp_rate=1_000_000, rolloff=0.35):
        gr.hier_block2.__init__(
            self,
            "QPSK_RX_block",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signaturev(2, 2, [gr.sizeof_gr_complex, gr.sizeof_char]),
        )

        sym_rate = samp_rate // sps
        ntaps = 15 * sps + 1

        rrc_taps = filter.firdes.root_raised_cosine(
            gain=1.0,
            sampling_freq=samp_rate,
            symbol_rate=sym_rate,
            alpha=rolloff,
            ntaps=ntaps
        )

        self.rrc_rx = filter.fir_filter_ccf(1, rrc_taps)
        self.agc = analog.agc2_cc(1e-3, 1e-4, 1.41421356, 1.0)

        self.clock_sync = digital.symbol_sync_cc(
            digital.TED_GARDNER,
            sps,
            0.001,
            1.0,
            1.0,
            1.5,
            1,
            digital.constellation_qpsk().base(),
            digital.IR_MMSE_8TAP
        )

        self.costas = digital.costas_loop_cc(0.0003, 4, False)
        self.copy = blocks.copy(gr.sizeof_gr_complex)

        self.const = digital.constellation_qpsk().base()
        self.decoder = digital.constellation_decoder_cb(self.const)
        self.sym2bit = digital.map_bb([0, 1, 3, 2])
        self.unpack = blocks.unpack_k_bits_bb(2)

        self.connect(self, self.rrc_rx, self.agc,
                     self.clock_sync, self.costas, self.copy)

        self.connect(self.copy, (self, 0))
        self.connect(self.costas, self.decoder, self.sym2bit, self.unpack, (self, 1))
