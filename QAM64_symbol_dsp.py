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
#  64QAM TX Block
# ---------------------------------------------------------
class QAM64_TX_block(gr.hier_block2):
    def __init__(self, sps=4, samp_rate=1_000_000, rolloff=0.35):
        gr.hier_block2.__init__(
            self,
            "QAM64_TX_block",
            gr.io_signature(1, 1, gr.sizeof_char),
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
        )

        sym_rate = samp_rate // sps
        ntaps = 15 * sps + 1

        # 使用相容性最佳的 64QAM 星座物件宣告
        self.const = digital.qam_constellation(64).base()
        points = self.const.points()

        # 將輸入符號索引 (0~63) 映射至複數星座點
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
#  64QAM RX Block (兼顧快鎖與低 EVM 的終極折衷版)
# ---------------------------------------------------------
class QAM64_RX_block(gr.hier_block2):
    def __init__(self, sps=4, samp_rate=1_000_000, rolloff=0.35):
        gr.hier_block2.__init__(
            self,
            "QAM64_RX_block",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signaturev(2, 2, [gr.sizeof_gr_complex, gr.sizeof_char]),
        )

        sym_rate = samp_rate // sps
        ntaps = 15 * sps + 1
        self.constellation_point = 64

        rrc_taps = filter.firdes.root_raised_cosine(
            gain=1.0,
            sampling_freq=samp_rate,
            symbol_rate=sym_rate,
            alpha=rolloff,
            ntaps=ntaps
        )

        self.rrc_rx = filter.fir_filter_ccf(1, rrc_taps)
        self.agc = analog.agc2_cc(1e-3, 1e-4, 1.0, 1.0)

        # 64QAM 星座物件
        self.const = digital.qam_constellation(64).base()

        # Symbol Synchronization
        self.clock_sync = digital.symbol_sync_cc(
            digital.TED_GARDNER,
            sps,
            0.001,
            1.0,
            1.0,
            1.5,
            1,
            self.const,
            digital.IR_MMSE_8TAP
        )

        # 關鍵折衷 1：CMA 保持 15 Taps + 0.00015 步長
        # (比 0.0005 乾淨很多，但比 0.00005 收斂速度快 3 倍)
        cma_alg = digital.adaptive_algorithm_cma(self.const, 0.00015, 1.0).base()
        self.equalizer = digital.linear_equalizer(
            15,
            sps,
            cma_alg,
            True
        )

        # 關鍵折衷 2：Costas Loop 設定為 0.00025
        # (主要靠 Costas 的強大抓力來提供「秒鎖感」，而不會把點位弄胖)
        self.costas = digital.costas_loop_cc(0.00025, 4, False)
        self.copy = blocks.copy(gr.sizeof_gr_complex)

        self.decoder = digital.constellation_decoder_cb(self.const)
        self.unpack = blocks.unpack_k_bits_bb(6)

        # 訊號串接
        self.connect(self, self.rrc_rx, self.agc,
                     self.clock_sync, self.equalizer, self.costas, self.copy)

        self.connect(self.copy, (self, 0))
        self.connect(self.costas, self.decoder, self.unpack, (self, 1))