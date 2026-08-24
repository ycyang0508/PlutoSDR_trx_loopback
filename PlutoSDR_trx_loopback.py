#!/usr/bin/env python3
import sys
import numpy as np
import adi
from PyQt5 import Qt
import sip
from gnuradio import gr, blocks, analog, digital, qtgui, filter, fft
from custom_blocks import *

class PlutoSDR_txrx_stream(gr.sync_block):
    def __init__(self, uri="ip:192.168.2.1", samp_rate=1_000_000,
                 tx_lo=915e6, rx_lo=915e6, buf_len=16384):

        gr.sync_block.__init__(
            self,
            name="pluto_txrx_stream",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )

        self.buf_len = buf_len
        self.set_output_multiple(buf_len)

        # 1. Pluto SDR 初始化
        self.sdr = adi.Pluto(uri)
        self.sdr.sample_rate = int(samp_rate)
        
        # 2. TX 硬體設定
        self.sdr.tx_lo = int(tx_lo)
        self.sdr.tx_hardwaregain_chan0 = -10   
        self.sdr.tx_buffer_size = buf_len
        self.sdr.tx_cyclic_buffer = False   

        # 3. RX 硬體設定
        self.sdr.rx_lo = int(rx_lo)
        self.sdr.gain_control_mode_chan0 = 'manual'
        self.sdr.rx_hardwaregain_chan0 = 10    
        self.sdr.rx_buffer_size = buf_len
        self.sdr.rx_rf_bandwidth = int(samp_rate * 2)

    def work(self, input_items, output_items):
        in_data = input_items[0]
        out_data = output_items[0]
        n_out = len(out_data)

        for i in range(0, n_out, self.buf_len):
            end = min(i + self.buf_len, n_out)
            
            # PlutoSDR Block 內部：將 [-1.0, +1.0] 的浮點數轉成硬體 DAC 範圍
            tx_chunk = in_data[i:end]
            if len(tx_chunk) > 0:
                self.sdr.tx(tx_chunk * (2**14))

            # 同步讀取 RX 資料並除回標準浮點數 [-1.0, +1.0]
            rx_chunk = self.sdr.rx()
            out_data[i:end] = rx_chunk[:(end - i)] / (2**14)

        return n_out


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