import sys
from gnuradio import analog
from gnuradio import blocks
from gnuradio import digital
from gnuradio import filter as grfilter
from gnuradio import gr
from gnuradio import qtgui
from gnuradio.filter import firdes
import numpy as np
import pmt
from PyQt5 import Qt
import sip
from gnuradio import channels

# ============================================================
# 1. 統一 Constellation 物件 (發送與接收共用)
# ============================================================
QPSK_CONST = digital.constellation_qpsk().base()
QPSK_POINTS = QPSK_CONST.points()

BARKER_13_RAW = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
BARKER_26_BITS = BARKER_13_RAW * 2


def barker_to_qpsk_symbols(barker_list):
  # 1 -> Index 0, -1 -> Index 3 (對角線 BPSK 映射)
  return [QPSK_POINTS[0] if val == 1 else QPSK_POINTS[3] for val in barker_list]


QPSK_PREAMBLE_SYMBOLS = barker_to_qpsk_symbols(BARKER_26_BITS)


# ============================================================
# 2. Header Parser (mark_delay = 0，微調 offset = -1)
# ============================================================
class qpsk_header_parser(gr.basic_block):
    def __init__(self, preamble_len_syms, header_len_bytes=4):
        gr.basic_block.__init__(self, name="qpsk_header_parser", in_sig=[np.complex64], out_sig=None)
        self.pre_len_syms = preamble_len_syms
        self.header_len_bytes = header_len_bytes
        self.header_len_syms = header_len_bytes * 4
        self.qpsk_const = QPSK_CONST
        # 預先準備好理想的 Preamble 複數陣列
        self.ref_preamble = np.array(QPSK_PREAMBLE_SYMBOLS, dtype=np.complex64)

    def general_work(self, input_items, output_items):
        in_iq = input_items[0]
        n = len(in_iq)
        if n == 0:
            return 0

        tags = self.get_tags_in_window(0, 0, n)
        for t in tags:
            if t.key == pmt.intern("corr_start"):
                corr_sym_idx = t.offset - self.nitems_read(0)

                # Preamble 與 Header 的起止位置
                pre_start = corr_sym_idx + 1
                pre_end = pre_start + self.pre_len_syms
                hdr_start = pre_end
                hdr_end = hdr_start + self.header_len_syms

                if pre_start < 0 or hdr_end > n:
                    continue

                # 1. 取得 Costas / Correlator 估計的粗相位
                phase_est = 0.0
                if pmt.is_dict(t.value):
                    phase_pmt = pmt.dict_ref(t.value, pmt.intern("phase_est"), pmt.PMT_NIL)
                    if not pmt.is_null(phase_pmt):
                        phase_est = pmt.to_double(phase_pmt)

                # 2. 先套用粗相位補償
                pre_iq = in_iq[pre_start:pre_end] * np.exp(-1j * phase_est)
                hdr_iq = in_iq[hdr_start:hdr_end] * np.exp(-1j * phase_est)

                # 3. 【關鍵：相位解模糊】利用 Preamble 與理想 Preamble 計算 4 個象限的點積能量
                # 計算四種可能旋轉角度 (0, 90, 180, 270 度) 下的相互關聯度
                rotations = [1.0, 1j, -1.0, -1j] # 對應 0, 90, 180, 270 度
                metric = [np.real(np.sum(pre_iq * np.conj(self.ref_preamble * rot))) for rot in rotations]
                best_rot_idx = np.argmax(metric) # 找出相相關度最高的旋轉相位
                best_rot = rotations[best_rot_idx]

                # 4. 將 Header 轉回正確的 0 度象限
                hdr_iq_corrected = hdr_iq * np.conj(best_rot)

                # 5. 硬判決與轉成 Byte
                hdr_syms = [self.qpsk_const.decision_maker(sample) for sample in hdr_iq_corrected]

                header_bytes = []
                for i in range(0, self.header_len_syms, 4):
                    byte_val = (int(hdr_syms[i]) << 6) | (int(hdr_syms[i+1]) << 4) | (int(hdr_syms[i+2]) << 2) | int(hdr_syms[i+3])
                    header_bytes.append(byte_val)

                phase_deg = np.degrees(phase_est)
                ambiguity_deg = best_rot_idx * 90
                print(f"[Header Parser] Phase: {phase_deg:4.1f}°, Ambiguity Fix: +{ambiguity_deg}° | Header: {[hex(b) for b in header_bytes]}")

        self.consume(0, n)
        return 0

# ============================================================
# 3. TX Block
# ============================================================
class tx_block(gr.hier_block2):

  def __init__(self, sps=4, alpha=0.35):
    gr.hier_block2.__init__(
        self,
        'tx_block',
        gr.io_signature(0, 0, 0),
        gr.io_signature(1, 1, gr.sizeof_gr_complex),
    )

    samp_rate = 100000
    sym_rate = samp_rate // sps
    ntaps = 11 * sps

    def bytes_to_qpsk_symbols(bl):
      syms = []
      for b in bl:
        for shift in (6, 4, 2, 0):
          idx = (b >> shift) & 0x03
          syms.append(QPSK_POINTS[idx])
      return syms

    # 1. Dummy
    dummy_syms = barker_to_qpsk_symbols([1, -1] * 32)
    # 2. Preamble
    preamble_syms = QPSK_PREAMBLE_SYMBOLS
    # 3. Header
    header_syms = bytes_to_qpsk_symbols([0x10, 0xAA, 0xCC, 0xEE])
    # 4. Payload    
    payload_bytes = np.random.randint(0, 256, 16, dtype=np.uint8).tolist()
    payload_syms = bytes_to_qpsk_symbols(payload_bytes)
    # 5. Zero Padding
    zeros_syms = [complex(0, 0)] * 64

    full_packet = (
        dummy_syms + preamble_syms + header_syms + payload_syms + zeros_syms
    )
    full_packet_np = np.array(full_packet, dtype=np.complex64)

    self.src = blocks.vector_source_c(full_packet_np.tolist(), repeat=True)

    rrc = firdes.root_raised_cosine(1.0, samp_rate, sym_rate, alpha, ntaps)
    self.rrc = grfilter.interp_fir_filter_ccf(sps, rrc)
    self.throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)

    self.connect(self.src, self.rrc, self.throttle, self)


# ============================================================
# 4. RX Block
# ============================================================
class rx_block(gr.hier_block2):

  def __init__(self, sps=4, alpha=0.35):
    gr.hier_block2.__init__(
        self,
        'rx_block',
        gr.io_signature(1, 1, gr.sizeof_gr_complex),
        gr.io_signature(0, 0, 0),
    )

    samp_rate = 100000
    sym_rate = samp_rate // sps
    ntaps = 11 * sps

    rrc = firdes.root_raised_cosine(1, samp_rate, sym_rate, alpha, ntaps)

    self.symbol_sync = digital.symbol_sync_cc(
        digital.TED_GARDNER,
        sps,
        0.0628,
        1.0,
        1.0,
        1.5,
        1,
        None,
        digital.IR_MMSE_8TAP,
        128,
        rrc,
    )

    self.gain_fix = blocks.multiply_const_cc(3.0)
    self.agc = analog.agc2_cc(1e-3, 1e-4, 1.5, 1.0)
    self.costas = digital.costas_loop_cc(0.0628, 4)

    preamble_symbols = np.array(QPSK_PREAMBLE_SYMBOLS, dtype=np.complex64)
    self.corr = digital.corr_est_cc(
        preamble_symbols.tolist(), sps=1, mark_delay=0, threshold=0.6
    )

    self.header_parser = qpsk_header_parser(
        preamble_len_syms=len(QPSK_PREAMBLE_SYMBOLS), header_len_bytes=4
    )
    self.qt_pre = qtgui.const_sink_c(256, 'Constellation Diagram', 1)

    self.connect(self, self.symbol_sync)
    self.connect(self.symbol_sync, self.gain_fix)
    self.connect(self.gain_fix, self.agc)
    self.connect(self.agc, self.costas)
    self.connect(self.costas, self.corr)
    self.connect(self.corr, self.header_parser)
    self.connect(self.costas, self.qt_pre)


# ============================================================
# 5. GUI Top Block
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

    self.channel = channels.channel_model(
        noise_voltage=0.05,      # AWGN noise level
        frequency_offset=0.0002, # CFO (carrier frequency offset)
        epsilon=1.0,             # timing offset (1.0 = no timing error)
        taps=[1.0+0j],           # multipath taps (可改成多路徑)
        noise_seed=42,           # 固定 seed 方便重現
        block_tags=False
    )

    self.tb.connect(self.tx, self.channel)
    self.tb.connect(self.channel, self.rx)

    layout = Qt.QVBoxLayout()
    self.setLayout(layout)
    layout.addWidget(sip.wrapinstance(self.rx.qt_pre.qwidget(), Qt.QWidget))

    self.tb.start()

  def closeEvent(self, event):
    self.tb.stop()
    self.tb.wait()
    event.accept()


if __name__ == '__main__':
  qapp = Qt.QApplication(sys.argv)
  win = top_gui()
  win.show()
  sys.exit(qapp.exec_())