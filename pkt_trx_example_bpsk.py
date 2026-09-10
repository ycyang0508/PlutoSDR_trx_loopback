from gnuradio import gr, blocks, digital, analog, filter as grfilter, qtgui
from gnuradio.filter import firdes
import numpy as np
import pmt
from PyQt5 import Qt
import sip
import sys

# ============================================================
# PREAMBLE (BPSK)
# ============================================================
DEFAULT_PREAMBLE_BITS = [
    1,1,0,0,0,1,1,1,0,1,0,0,1,0,1,1,
    1,0,0,1,1,1,0,1,1,0,0,0,0,1,0,1
]

# ============================================================
# Header Parser (BPSK)
# ============================================================
class bpsk_header_parser(gr.basic_block):
    def __init__(self, preamble_bits, header_len_bytes=4):
        gr.basic_block.__init__(
            self,
            name="bpsk_header_parser",
            in_sig=[np.uint8],
            out_sig=None
        )

        self.preamble = np.array(preamble_bits, dtype=np.uint8)
        self.pre_len = len(self.preamble)
        self.header_len_bits = header_len_bytes * 8

    def general_work(self, input_items, output_items):
        bits = input_items[0]
        n = len(bits)
        if n == 0:
            return 0

        tags = self.get_tags_in_window(0, 0, n)

        for t in tags:
            if t.key == pmt.intern("corr_start"):
                corr_bit = t.offset - self.nitems_read(0)
                pos = corr_bit + 1

                if pos < 0 or pos + self.pre_len > n:
                    continue

                window = bits[pos : pos + self.pre_len]
                score = np.sum(window == self.preamble)

                if score != self.pre_len:
                    print("[Header Parser] Preamble mismatch, score:", score)
                    continue

                hdr_start = pos + self.pre_len
                hdr_end   = hdr_start + self.header_len_bits

                if hdr_end > n:
                    continue

                header_bits = bits[hdr_start:hdr_end]

                header_bytes = []
                for i in range(0, self.header_len_bits, 8):
                    val = 0
                    for j in range(8):
                        val = (val << 1) | int(header_bits[i+j])
                    header_bytes.append(val)

                print(f"[Header Parser] 解出 Header: {[hex(b) for b in header_bytes]}")

        self.consume(0, n)
        return 0

# ============================================================
# TX (全部 BPSK)
# ============================================================
class tx_block(gr.hier_block2):
    def __init__(self, sps=4, alpha=0.35):
        gr.hier_block2.__init__(
            self,
            "tx_block",
            gr.io_signature(0,0,0),
            gr.io_signature(1,1,gr.sizeof_gr_complex)
        )

        samp_rate = 100000
        sym_rate = samp_rate // sps
        ntaps = 11 * sps

        pre_bits = DEFAULT_PREAMBLE_BITS
        header_bytes = [0x10, 0xAA, 0xBB, 0xCC]
        payload_bytes = list(range(0x01, 0x11))

        def bytes_to_bits(bl):
            out = []
            for b in bl:
                for i in range(8):
                    out.append((b >> (7 - i)) & 1)
            return out

        header_bits = bytes_to_bits(header_bytes)
        payload_bits = bytes_to_bits(payload_bytes)

        bpsk_points = digital.constellation_bpsk().base().points()

        self.src_pre = blocks.vector_source_b(pre_bits, repeat=True)
        self.src_hdr = blocks.vector_source_b(header_bits, repeat=True)
        self.src_pay = blocks.vector_source_b(payload_bits, repeat=True)

        self.pre_map = digital.chunks_to_symbols_bc(bpsk_points, 1)
        self.hdr_map = digital.chunks_to_symbols_bc(bpsk_points, 1)
        self.pay_map = digital.chunks_to_symbols_bc(bpsk_points, 1)

        
        self.mux = blocks.stream_mux(
            gr.sizeof_gr_complex,
            [len(pre_bits), len(header_bits), len(payload_bits)]
        )

        rrc = firdes.root_raised_cosine(1.0, samp_rate, sym_rate, alpha, ntaps)
        self.rrc = grfilter.interp_fir_filter_ccf(sps, rrc)
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)
        
        self.connect(self.src_pre, self.pre_map)
        self.connect(self.src_hdr, self.hdr_map)
        self.connect(self.src_pay, self.pay_map)
        
        self.connect((self.pre_map, 0), (self.mux, 0))
        self.connect((self.hdr_map, 0), (self.mux, 1))
        self.connect((self.pay_map, 0), (self.mux, 2))

        self.connect(self.mux, self.rrc, self.throttle)
        self.connect(self.throttle, self)

# ============================================================
# RX (全部 BPSK)
# ============================================================
class rx_block(gr.hier_block2):
    def __init__(self, sps=4, alpha=0.35):
        gr.hier_block2.__init__(
            self,
            "rx_block",
            gr.io_signature(1,1,gr.sizeof_gr_complex),
            gr.io_signature(0,0,0)
        )

        samp_rate = 100000
        sym_rate = samp_rate // sps
        ntaps = 11 * sps

        # ----------------------------------------
        # Throttle + AGC
        # ----------------------------------------
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)
        

        # ----------------------------------------
        # RRC taps (給 PFB 使用)
        # ----------------------------------------
        rrc = firdes.root_raised_cosine(
            1, samp_rate, sym_rate, alpha, ntaps
        )

        # ----------------------------------------
        # ⭐ PFB Clock Sync (取代 clock_recovery_mm_cc)
        # ----------------------------------------
        self.symbol_sync = digital.symbol_sync_cc(
                                                  digital.TED_GARDNER,  # TED 類型
                                                  sps,                  # samples per symbol
                                                  0.0628,               # loop bandwidth
                                                  1.0,                  # damping factor
                                                  1.0,                  # TED gain
                                                  1.5,                  # max deviation
                                                  1,                    # osps
                                                  None,                 # slicer (BPSK 可用 None)
                                                  digital.IR_MMSE_8TAP, # interpolation type
                                                  128,                  # number of filters
                                                  rrc                   # RRC taps
                                              )

        self.gain_fix = blocks.multiply_const_cc(3.0)
        self.agc = analog.agc2_cc(1e-3, 1e-4, 1, 1.0)

        # ----------------------------------------
        # Costas Loop (BPSK → order=2)
        # ----------------------------------------
        self.costas = digital.costas_loop_cc(0.0628, 4)
        self.phase_rot = blocks.multiply_const_cc(np.exp(-1j * np.pi/4))

        # ----------------------------------------
        # Correlation Estimator (產生 corr_start tag)
        # ----------------------------------------
        pre_bpsk = np.array(
            [1 if b==1 else -1 for b in DEFAULT_PREAMBLE_BITS],
            dtype=np.complex64
        ).tolist()

        self.corr = digital.corr_est_cc(pre_bpsk, 1, 0, 0.9)

        # ----------------------------------------
        # BPSK Decoder
        # ----------------------------------------
        self.bpsk_const = digital.constellation_bpsk().base()
        self.decoder = digital.constellation_decoder_cb(self.bpsk_const)

        # ----------------------------------------
        # Header Parser
        # ----------------------------------------
        self.header_parser = bpsk_header_parser(DEFAULT_PREAMBLE_BITS, header_len_bytes=4)

        # ----------------------------------------
        # GUI Sink
        # ----------------------------------------
        self.qt_pre = qtgui.const_sink_c(256, "PREAMBLE", 1)
        self.qt_debug = qtgui.time_sink_c(256,samp_rate, "agc", 1)
        self.qt_debug1 = qtgui.time_sink_c(256,samp_rate, "symbol_sync", 1)

        # ----------------------------------------
        # ⭐ 新的 RX 連線架構
        # ----------------------------------------
        self.connect(self, self.throttle)
        self.connect(self.throttle, self.symbol_sync)
        self.connect(self.symbol_sync, self.gain_fix)
        self.connect(self.gain_fix, self.agc)
        self.connect(self.agc, self.costas)
        self.connect(self.costas, self.phase_rot)
        self.connect(self.phase_rot, self.corr)
        self.connect(self.corr, self.decoder, self.header_parser)

        # GUI
        self.connect(self.phase_rot, self.qt_pre)
        self.connect(self.agc, self.qt_debug)
        self.connect(self.symbol_sync, self.qt_debug1)


# ============================================================
# GUI Top Block
# ============================================================
class top_gui(Qt.QWidget):
    def __init__(self):
        super().__init__()

        self.tb = gr.top_block()

        sps = 4
        alpha = 0.35

        self.tx = tx_block(sps, alpha)
        self.ch = blocks.copy(gr.sizeof_gr_complex)
        self.rx = rx_block(sps, alpha)

        self.tb.connect(self.tx, self.ch)
        self.tb.connect(self.ch, self.rx)

        layout = Qt.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(sip.wrapinstance(self.rx.qt_pre.qwidget(), Qt.QWidget))
        #layout.addWidget(sip.wrapinstance(self.rx.qt_debug.qwidget(), Qt.QWidget))
        #layout.addWidget(sip.wrapinstance(self.rx.qt_debug1.qwidget(), Qt.QWidget))

        self.tb.start()

    def closeEvent(self, event):
        self.tb.stop()
        self.tb.wait()
        event.accept()

if __name__ == "__main__":
    qapp = Qt.QApplication(sys.argv)
    win = top_gui()
    win.show()
    qapp.exec_()