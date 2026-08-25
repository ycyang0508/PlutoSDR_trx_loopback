#!/usr/bin/env python3
import sys
import numpy as np
import adi
from PyQt5 import Qt
import sip
from gnuradio import gr, blocks, analog, digital, qtgui, filter, fft
from custom_blocks import *

import numpy as np
from gnuradio import gr, blocks
import adi


class qpsk_cable_demo(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self)

        # === 系統基本參數 ===
        samp_rate = 1_000_000   
        sps = 4
        sym_rate = samp_rate // sps
        rolloff = 0.35
        ntaps = 11 * sps + 1
        buf_len = 8192

        # 1. 計算 RRC 濾波器 Taps
        rrc_taps = filter.firdes.root_raised_cosine(
            gain=1.0, sampling_freq=samp_rate, symbol_rate=sym_rate, alpha=rolloff, ntaps=ntaps
        )

        # 2. QPSK 星座圖定義
        const = digital.constellation_qpsk().base()
        points = const.points()

        # 3. TX 訊號鏈路建構
        n_symbols = 200000
        rnd = np.random.randint(0, 4, n_symbols).tolist()
        self.src = blocks.vector_source_b(rnd, repeat=True)

        # 符號映射 Mapping
        self.mapper = digital.chunks_to_symbols_bc(points, 1)

        # TX Pulse Shaping (RRC 插值濾波器，直接輸出標準振幅範疇)
        self.rrc_tx = filter.interp_fir_filter_ccf(sps, rrc_taps)

        # 4. 實體化 Pluto Transceiver Block

        self.pluto = PlutoSDR_txrx_stream(
            uri="ip:192.168.2.1",
            samp_rate=samp_rate,
            tx_lo=915e6,
            rx_lo=915e6,
            buf_len=buf_len
        )
        #self.pluto = FakePlutoSDR(
        #    fifo_size=buf_len * 4,
        #    samp_rate=samp_rate,
        #    freq_offset_hz=500.0
        #)


        # 5. UI 視覺化視窗
        self.freq_sink = qtgui.freq_sink_c(2048, fft.window.WIN_HAMMING, 0, samp_rate, "Rx Spectrum (1MHz)")
        self.freq_win = sip.wrapinstance(self.freq_sink.qwidget(), Qt.QWidget)

        self.const_sink = qtgui.const_sink_c(4096, "Cable Loopback QPSK (1MHz)", 1)
        self.const_sink.set_x_axis(-2.0, 2.0)
        self.const_sink.set_y_axis(-2.0, 2.0)
        self.const_win = sip.wrapinstance(self.const_sink.qwidget(), Qt.QWidget)

        # 6. RX 解調鏈路元件
        self.dc_block = filter.dc_blocker_cc(128, True)
        self.agc = analog.agc2_cc(1e-2, 1e-3, 1.0, 1.0)
        self.rrc_rx = filter.fir_filter_ccf(1, rrc_taps)
        
        self.clock_sync = digital.symbol_sync_cc(
            digital.TED_GARDNER,
            sps,
            0.015,
            1.0,
            1.0,
            1.5,
            1,
            digital.constellation_qpsk().base(),
            digital.IR_MMSE_8TAP
        )
        self.costas = digital.costas_loop_cc(0.015, 4, False)

        # 7. 連接 GNU Radio 鏈路
        # [TX 鏈路]: Bits Source -> Mapper -> Interp FIR -> Pluto Block
        self.connect(self.src, self.mapper, self.rrc_tx, self.pluto)

        # [RX 鏈路]: Pluto Block -> Spectrum & RX DSP
        self.connect(self.pluto, self.freq_sink)
        self.connect(self.pluto, self.dc_block, self.agc, self.rrc_rx, self.clock_sync, self.costas, self.const_sink)

def run_demo():
    app = Qt.QApplication(sys.argv)
    tb = qpsk_cable_demo()

    win = Qt.QWidget()
    layout = Qt.QVBoxLayout(win)
    layout.addWidget(tb.freq_win)
    layout.addWidget(tb.const_win)
    win.show()

    tb.start()
    app.exec_()
    tb.stop()
    tb.wait()

if __name__ == "__main__":
    run_demo()