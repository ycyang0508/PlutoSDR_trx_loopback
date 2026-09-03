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
#  256QAM TX Block (Char Input)
# ---------------------------------------------------------
class QAM256_TX_block(gr.hier_block2):
    def __init__(self, sps=4, samp_rate=1_000_000, rolloff=0.35):
        gr.hier_block2.__init__(
            self,
            "QAM256_TX_block",
            gr.io_signature(1, 1, gr.sizeof_char),
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
        )

        sym_rate = samp_rate // sps
        ntaps = 15 * sps + 1

        self.const = digital.qam_constellation(256).base()
        points = self.const.points()

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
#  256QAM RX Block (Char Output, 支援抗大 CFO & 相位鎖定)
# ---------------------------------------------------------
class QAM256_RX_block(gr.hier_block2):
    def __init__(self, sps=4, samp_rate=1_000_000, rolloff=0.35):
        gr.hier_block2.__init__(
            self,
            "QAM256_RX_block",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signaturev(2, 2, [gr.sizeof_gr_complex, gr.sizeof_char]),
        )

        sym_rate = samp_rate // sps
        ntaps = 15 * sps + 1
        nfilters = 32
        self.constellation_point = 256

        # 1. 建立標準 256QAM 星座圖物件
        self.const = digital.qam_constellation(256).base()

        # 2. 自動增益控制 (AGC)
        self.agc = analog.agc2_cc(1e-2, 1e-3, 1.0, 1.0)

        # 3. Polyphase RRC Filter Taps
        rrc_taps = filter.firdes.root_raised_cosine(
            gain=nfilters,  # Polyphase 增益補償
            sampling_freq=samp_rate * nfilters,
            symbol_rate=sym_rate,
            alpha=rolloff,
            ntaps=ntaps * nfilters
        )

        # 4. Symbol Synchronization (1 SPS 重採樣 & 時脈同步)
        self.clock_sync = digital.symbol_sync_cc(
            detector_type=digital.TED_MOD_MUELLER_AND_MULLER,
            sps=sps,
            loop_bw=0.0001,                 # 小頻寬防止採樣點晃動
            damping_factor=0.707,
            ted_gain=1.0,
            max_deviation=1.0,
            osps=1,
            slicer=self.const,              # Decision-Directed 採樣
            interp_type=digital.IR_PFB_MF,
            n_filters=nfilters,
            taps=rrc_taps
        )

        # 5. 精密單級 Costas Loop (載波鎖頻與鎖相)
        # 【核心修正】：移除會扯爆星座圖的 0.005 粗鎖環路
        # 設定為 0.0001 的超低 Loop BW，既能抓住殘留頻偏，又能完全凝固 256QAM 相位
        self.costas = digital.costas_loop_cc(0.0001, 4, False)
        
        self.copy = blocks.copy(gr.sizeof_gr_complex)

        # 6. 解碼與 Unpack
        self.decoder = digital.constellation_decoder_cb(self.const)
        self.unpack = blocks.unpack_k_bits_bb(8)

        # 7. 乾淨訊號鏈串接: AGC -> Symbol Sync -> Costas Loop
        self.connect(self, self.agc, self.clock_sync, self.costas, self.copy)
        self.connect(self.copy, (self, 0))
        self.connect(self.costas, self.decoder, self.unpack, (self, 1))