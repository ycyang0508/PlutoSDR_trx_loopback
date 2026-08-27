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
#  16QAM TX Block
# ---------------------------------------------------------
class QAM16_TX_block(gr.hier_block2):
    def __init__(self, sps=4, samp_rate=1_000_000, rolloff=0.35):
        gr.hier_block2.__init__(
            self,
            "QAM16_TX_block",
            gr.io_signature(1, 1, gr.sizeof_char),  # 輸入為 byte/char
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
        )

        sym_rate = samp_rate // sps
        ntaps = 15 * sps + 1        

        # 建立 16QAM Constellation 物件
        self.const = digital.constellation_16qam().base()
        points = self.const.points()

        # 將輸入的 symbol index (0~15) 映射至星座點複數值
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
#  16QAM RX Block（輸出 symbols + bits）
# ---------------------------------------------------------
class QAM16_RX_block(gr.hier_block2):
    def __init__(self, sps=4, samp_rate=1_000_000, rolloff=0.35):
        gr.hier_block2.__init__(
            self,
            "QAM16_RX_block",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signaturev(2, 2, [gr.sizeof_gr_complex, gr.sizeof_char]),
        )

        sym_rate = samp_rate // sps
        ntaps = 15 * sps + 1
        self.constellation_point = 16

        rrc_taps = filter.firdes.root_raised_cosine(
            gain=1.0,
            sampling_freq=samp_rate,
            symbol_rate=sym_rate,
            alpha=rolloff,
            ntaps=ntaps
        )

        self.rrc_rx = filter.fir_filter_ccf(1, rrc_taps)
        
        # AGC: 16QAM 的平均能量 (RMS) 為 1.0，但峰值會大於 1
        self.agc = analog.agc2_cc(1e-3, 1e-4, 1.0, 1.0)

        # 16QAM 星座物件
        self.const = digital.constellation_16qam().base()

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

        # Costas Loop: 16QAM 為 4-fold 相位對稱 (Order = 4)
        self.costas = digital.costas_loop_cc(0.0003, 4, False)
        self.copy = blocks.copy(gr.sizeof_gr_complex)

        # 解調與解包
        self.decoder = digital.constellation_decoder_cb(self.const)
        
        # 16QAM 每個 symbol 代表 4 個 bits (4 bits unpack)
        self.unpack = blocks.unpack_k_bits_bb(4)

        self.connect(self, self.rrc_rx, self.agc,
                     self.clock_sync, self.costas, self.copy)

        # Port 0: 輸出解調後的複數 Symbol 訊號 (供觀察 Constellation 圖)
        self.connect(self.copy, (self, 0))
        
        # Port 1: 輸出解包後的 Bit 流 (0 或 1)
        self.connect(self.costas, self.decoder, self.unpack, (self, 1))