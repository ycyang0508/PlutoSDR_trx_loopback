import sys
import numpy as np
import adi
from PyQt5 import Qt
import sip
from gnuradio import gr, analog, qtgui, fft,blocks
from PlutoSDR_trx import *
from custom_blocks import *

class pluto_loopback_demo(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self)

        samp_rate = 1_000_000
        tone_freq = samp_rate / 6 * 1
        lo_freq = 2_400_000_000
        buf_len = 8192
        amp_max = 0.999999
        amp_min = 1.0/2**14

        # Tone Source
        self.tone = analog.sig_source_c(
            samp_rate,
            analog.GR_SIN_WAVE,
            tone_freq,
            amp_max
        )

        # Pluto TX/RX
        self.pluto = PlutoSDR_txrx(
            samp_rate=samp_rate,
            tx_lo=lo_freq,
            rx_lo=lo_freq,
            buf_len=buf_len
        )
        
        self.tx_q15 = blocks.multiply_const_cc( (2**11) + 0j) # float to Q5.11
        self.rx_q15 = blocks.multiply_const_cc( (1.0/(2**9.3)) + 0j)

        # FFT
        self.fft = qtgui.freq_sink_c(
            buf_len,
            fft.window.WIN_HAMMING,
            0,
            samp_rate,
            "Pluto SDR Loopback FFT",
            1
        )
        self.fft_win = sip.wrapinstance(self.fft.qwidget(), Qt.QWidget)

        # Connect
        #self.connect(self.tone, self.pluto)
        #self.connect(self.pluto, self.fft)

        self.connect(self.tone, self.tx_q15)
        self.connect(self.tx_q15, self.pluto)

        self.connect(self.pluto,self.rx_q15)
        self.connect(self.rx_q15, self.fft)


        #time domain
        self.pure_time = qtgui.time_sink_c(
            buf_len,          # buffer size
            samp_rate,     # sample rate
            "Time Domain", # title
           1              # number of inputs
        )
        self.connect(self.rx_q15, self.pure_time)        
        self.time_win = sip.wrapinstance(self.pure_time.qwidget(), Qt.QWidget)
              

if __name__ == "__main__":
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
