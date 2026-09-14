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
#  16QAM RX Block（新增 LMS Equalizer）
# ---------------------------------------------------------
class QAM16_RX_block(gr.hier_block2):

    def __init__(self, sps=4, samp_rate=1_000_000, rolloff=0.35, eq_taps=15, eq_gain=0.001):
        gr.hier_block2.__init__(
            self,
            "QAM16_RX_block",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signaturev(2, 2, [gr.sizeof_gr_complex, gr.sizeof_char]),
        )

        sym_rate = samp_rate // sps
        ntaps = 15 * sps + 1
        self.constellation_point = 16

        # 1. RRC Filter
        rrc_taps = filter.firdes.root_raised_cosine(
            gain=1.0,
            sampling_freq=samp_rate,
            symbol_rate=sym_rate,
            alpha=rolloff,
            ntaps=ntaps,
        )
        self.rrc_rx = filter.fir_filter_ccf(1, rrc_taps)

        # 2. AGC
        self.agc = analog.agc2_cc(1e-3, 1e-4, 1.0, 1.0)

        # 3. Constellation 物件
        self.const = digital.constellation_16qam().base()

        # 4. Symbol Sync (Gardner)
        self.clock_sync = digital.symbol_sync_cc(
            digital.TED_GARDNER,
            sps,
            0.001,
            1.0,
            1.0,
            1.5,
            1,
            self.const,
            digital.IR_MMSE_8TAP,
        )

        # 5. [NEW] Linear Equalizer (LMS Algorithm)
        # 對 16QAM 而言，LMS 比 CMA 更適合用於多振幅星座圖的決策導向等化
        #self.eq_alg = digital.adaptive_algorithm_lms(self.const, eq_gain).base()
        self.eq_alg = digital.adaptive_algorithm_cma(self.const, eq_gain,1.0)
        self.eq = digital.linear_equalizer(
            num_taps=eq_taps,
            sps=1,  # Clock Sync 輸出已降至 1 sps
            alg=self.eq_alg,
            adapt_after_training=True
        )

        self.costas = digital.costas_loop_cc(0.01, 4)        
        
        # 6. CFO + phase + decision → symbol index (byte)
        self.cfo_sync = digital.constellation_receiver_cb(
            constellation=self.const,
            loop_bw=0.02,
            fmin=-0.05,
            fmax=0.05
        )
        

        # 7. 星座點轉換與 Bit 解包
        self.mapper_vis = digital.chunks_to_symbols_bc(self.const.points(), 1)
        self.unpack = blocks.unpack_k_bits_bb(4)
        self.copy = blocks.copy(gr.sizeof_gr_complex)

        # ---------------------------------------------------------
        # 主訊號流連線：
        # Input -> RRC -> AGC -> Clock Sync -> [EQ] -> CFO Sync
        # ---------------------------------------------------------
        self.connect(self, self.rrc_rx, self.agc, self.clock_sync, self.eq, self.costas, self.cfo_sync)

        # 星座圖輸出 (decision 後的點)
        #self.connect(self.cfo_sync, self.mapper_vis)
        self.connect(self.costas, self.copy, (self, 0))

        # Bits 輸出 (symbol index -> 4 bits)
        self.connect(self.cfo_sync, self.unpack, (self, 1))