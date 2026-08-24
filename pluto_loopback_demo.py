import sys
import numpy as np
import adi
from PyQt5 import Qt
import sip
from gnuradio import gr, analog, qtgui, fft,blocks,digital,blocks, filter, qtgui
from PlutoSDR_trx import *
from custom_blocks import *
import inspect
import gnuradio.digital as digital
#print(inspect.signature(digital.digital_python.pfb_clock_sync_ccf))
#help(digital.digital_python.pfb_clock_sync_ccf)


class pluto_loopback_demo(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self)

        self.samp_rate = 1_000_000
        tone_freq = self.samp_rate / 6 * 1
        lo_freq = 2_400_000_000
        buf_len = 8192
        amp_max = 0.999999
        amp_min = 1.0/2**14
        

        # Tone Source
        self.tone = analog.sig_source_c(
            self.samp_rate,
            analog.GR_SIN_WAVE,
            tone_freq,
            amp_max
        )

        # Pluto TX/RX
        self.pluto = PlutoSDR_txrx(
            samp_rate=self.samp_rate,
            tx_lo=lo_freq,
            rx_lo=lo_freq,
            buf_len=buf_len
        )
        
        self.tx_q15 = blocks.multiply_const_cc( (2**11) + 0j) # float to Q5.11
        self.rx_q15 = blocks.multiply_const_cc( (1.0/(2**9.3)) + 0j)
                

        # 放大或縮小搜尋範圍，以涵蓋你的目標頻率 (約 167.6 kHz)
        # 186.6 kHz 對應的 radians/sample 大約是 2 * pi * (186.6e3 / samp_rate)
        target_norm_freq = 2 * np.pi * (tone_freq / self.samp_rate)
        
        loop_bw   = 2 * np.pi * (5e3 / self.samp_rate)     # 5 kHz loop bandwidth
        max_freq  = target_norm_freq + 2 * np.pi * (50e3 / self.samp_rate)   # 上限
        min_freq  = target_norm_freq - 2 * np.pi * (50e3 / self.samp_rate)   # 下限

        # 改用 pll_carriertracking_cc，它會直接輸出把頻偏/載波移回基頻（DC）的訊號
        self.pll = analog.pll_carriertracking_cc(
            loop_bw,
            max_freq,
            min_freq
        )        


        # FFT
        self.fft = qtgui.freq_sink_c(
            buf_len,
            fft.window.WIN_HAMMING,
            0,
            self.samp_rate,
            "Pluto SDR Loopback FFT",
            2
        )
        self.fft_win = sip.wrapinstance(self.fft.qwidget(), Qt.QWidget)

        #time domain
        self.pure_time = qtgui.time_sink_c(
            buf_len,          # buffer size
            self.samp_rate,     # sample rate
            "Time Domain", # title
           1              # number of inputs
        )
        self.time_win = sip.wrapinstance(self.pure_time.qwidget(), Qt.QWidget)

        # Connect
        self.connect(self.tone, self.tx_q15)
        self.connect(self.tx_q15, self.pluto)
        self.connect(self.pluto,self.rx_q15)              
        
        self.connect(self.rx_q15, self.pure_time)

        
        # 直接將加了偏移的訊號丟進 Carrier Tracking PLL
        # 該 PLL 會自動把頻偏扣除，輸出對準 0 Hz（DC）的訊號
        self.connect(self.rx_q15, self.pll)


        # FFT out
        self.connect(self.rx_q15, (self.fft,0))        
        self.connect(self.pll,    (self.fft,1))              # output to FFT


        # --- PLL frequency monitor ---
        self.last_phase = 0
        self.monitor_timer = Qt.QTimer()
        self.monitor_timer.setInterval(100)   # 每 100 ms
        self.monitor_timer.timeout.connect(self.monitor_pll_freq)
        self.monitor_timer.start()
    
        
    def monitor_pll_freq(self):
        # get_frequency() 直接回傳 radians/sample
        norm_freq = self.pll.get_frequency()
        
        # 轉換成 Hz
        freq = (norm_freq / (2 * np.pi)) * self.samp_rate
        print(f"PLL NCO frequency estimate: {freq:.2f} Hz")        
        
def pluto_loopback_demo_tb() -> None:
    app = Qt.QApplication(sys.argv)
    tb = pluto_loopback_demo()

    win = Qt.QWidget()
    layout = Qt.QVBoxLayout(win)
    layout.addWidget(tb.fft_win)
    layout.addWidget(tb.time_win)
    win.show()

    tb.start()
    app.exec_()
    tb.stop()
    tb.wait()