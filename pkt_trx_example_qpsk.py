#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full QPSK demo (Dynamic Packet Version with Equalizer & Multi-path ISI Channel):
- TX: Dynamic Packet Generator
- channel_model (with Multipath ISI)
- RX: symbol_sync -> AGC1 -> Linear Equalizer (CMA) -> AGC2 -> Costas -> corr_est_cc -> header strip -> decoder -> parser
"""
import sys
import time
from gnuradio import analog
from gnuradio import blocks
from gnuradio import digital
from gnuradio import filter as grfilter
from gnuradio import gr
from gnuradio import qtgui
from gnuradio.filter import firdes
from gnuradio import channels
import numpy as np
import pmt
from PyQt5 import Qt
import sip

# ============================================================
# CRC-16 (IBM) for payload
# ============================================================
def crc16_ibm(data_bytes):
    crc = 0xFFFF
    for b in data_bytes:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF

# ============================================================
# 1. Constellation & Symbols Setup
# ============================================================
QPSK_CONST = digital.constellation_qpsk().base()
QPSK_POINTS = QPSK_CONST.points()

BARKER_13_RAW = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
BARKER_26_BITS = BARKER_13_RAW * 2

def barker_to_qpsk_symbols(barker_list):
    return [QPSK_POINTS[0] if v == 1 else QPSK_POINTS[3] for v in barker_list]

QPSK_PREAMBLE_SYMBOLS = barker_to_qpsk_symbols(BARKER_26_BITS)

def bytes_to_symidx(bl):
    out = []
    for b in bl:
        for shift in (6, 4, 2, 0):
            out.append((b >> shift) & 0x03)
    return out

# ============================================================
# 2. Dynamic TX Packet Source Block
# ============================================================
class dynamic_packet_generator(gr.sync_block):
    def __init__(self, min_payload_len=8, max_payload_len=32):
        gr.sync_block.__init__(
            self,
            name="dynamic_packet_generator",
            in_sig=None,
            out_sig=[np.complex64]
        )
        self.min_len = min_payload_len
        self.max_len = max_payload_len
        self.seq_num = 0

        self.dummy_syms = barker_to_qpsk_symbols([1, -1] * 32)
        self.preamble_syms = QPSK_PREAMBLE_SYMBOLS
        self.zeros_syms = [0+0j] * 64

        self.buffer = np.array([], dtype=np.complex64)

    def _generate_next_packet(self):
        payload_len = np.random.randint(self.min_len, self.max_len + 1)
        payload_bytes = np.random.randint(0, 256, payload_len, dtype=np.uint8).tolist()
        
        header_bytes = [payload_len, self.seq_num, 0xAA, 0xCC]
        
        crc_val = crc16_ibm(payload_bytes)
        crc_bytes = [(crc_val >> 8) & 0xFF, crc_val & 0xFF]
        
        hdr_syms = [QPSK_POINTS[idx] for idx in bytes_to_symidx(header_bytes)]
        pay_syms = [QPSK_POINTS[idx] for idx in bytes_to_symidx(payload_bytes + crc_bytes)]

        packet_syms = (
            self.dummy_syms + 
            self.preamble_syms + 
            hdr_syms + 
            pay_syms + 
            self.zeros_syms
        )

        self.seq_num = (self.seq_num + 1) % 256
        return np.array(packet_syms, dtype=np.complex64)

    def work(self, input_items, output_items):
        out = output_items[0]
        n_out = len(out)

        while len(self.buffer) < n_out:
            new_pkt = self._generate_next_packet()
            self.buffer = np.concatenate((self.buffer, new_pkt))

        out[:] = self.buffer[:n_out]
        self.buffer = self.buffer[n_out:]
        return n_out

class pkt_tx_QPSK(gr.hier_block2):
    def __init__(self, sps=4, samp_rate=1_000_000, alpha=0.35):
        gr.hier_block2.__init__(
            self,
            "tx_block",
            gr.io_signature(0,0,0),
            gr.io_signature(1,1,gr.sizeof_gr_complex)
        )

        samp_rate = samp_rate
        sym_rate = samp_rate // sps
        ntaps = 15 * sps + 1

        self.pkt_gen = dynamic_packet_generator(min_payload_len=8, max_payload_len=32)
        rrc = firdes.root_raised_cosine(1.0, samp_rate, sym_rate, alpha, ntaps)
        self.rrc = grfilter.interp_fir_filter_ccf(sps, rrc)
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)

        self.connect(self.pkt_gen, self.rrc, self.throttle, self)

# ============================================================
# 3. Header strip + phase correction block
# ============================================================
class qpsk_header_strip_with_phase(gr.basic_block):
    def __init__(self, preamble_len_syms, header_len_bytes=4, max_payload_bytes=256):
        gr.basic_block.__init__(
            self,
            name="qpsk_header_strip_with_phase",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self.pre_len_syms = int(preamble_len_syms)
        self.header_len_bytes = int(header_len_bytes)
        self.header_len_syms = self.header_len_bytes * 4
        self.qpsk_const = QPSK_CONST
        self.ref_preamble = np.array(QPSK_PREAMBLE_SYMBOLS, dtype=np.complex64)
        self._rots = [1.0, 1j, -1.0, -1j]

        self.max_payload_bytes = int(max_payload_bytes)
        self.max_payload_syms = (self.max_payload_bytes + 2) * 4
        self.max_packet_samples = 1 + self.pre_len_syms + self.header_len_syms + self.max_payload_syms
        self.set_output_multiple(self.max_packet_samples)

    def forecast(self, noutput_items, ninputs):
        need = self.max_packet_samples
        return [need] * ninputs

    def _get_phase_est_from_tag(self, tag):
        phase_est = 0.0
        if pmt.is_dict(tag.value):
            phase_pmt = pmt.dict_ref(tag.value, pmt.intern("phase_est"), pmt.PMT_NIL)
            if not pmt.is_null(phase_pmt):
                phase_est = pmt.to_double(phase_pmt)
        return phase_est

    def _resolve_ambiguity(self, pre_iq):
        rotations = self._rots
        metric = [np.real(np.sum(pre_iq * np.conj(self.ref_preamble * rot))) for rot in rotations]
        best_rot_idx = int(np.argmax(metric))
        return best_rot_idx, rotations[best_rot_idx]

    def general_work(self, input_items, output_items):
        in_iq = input_items[0]
        out_iq = output_items[0]

        n_in = len(in_iq)
        n_out_avail = len(out_iq)
        if n_in == 0 or n_out_avail == 0:
            return 0

        tags = self.get_tags_in_window(0, 0, n_in)
        corr_tags = [t for t in tags if t.key == pmt.intern("corr_start")]
        corr_tags.sort(key=lambda x: int(x.offset))

        n_read_abs = self.nitems_read(0)
        if not corr_tags:
            write_len = min(n_in, n_out_avail)
            out_iq[:write_len] = in_iq[:write_len]
            self.consume(0, write_len)
            return write_len

        in_pos = 0
        out_pos = 0        
        for t in corr_tags:
            rel_idx = int(t.offset - n_read_abs)

            if rel_idx < in_pos:
                continue

            pre_start = rel_idx + 1
            pre_end = pre_start + self.pre_len_syms
            hdr_start = pre_end
            hdr_end = hdr_start + self.header_len_syms

            if hdr_end > n_in:
                break

            phase_est = self._get_phase_est_from_tag(t)
            pre_iq = in_iq[pre_start:pre_end] * np.exp(-1j * phase_est)
            best_rot_idx, best_rot = self._resolve_ambiguity(pre_iq)

            hdr_iq = in_iq[hdr_start:hdr_end] * np.exp(-1j * phase_est) * np.conj(best_rot)
            hdr_syms = [self.qpsk_const.decision_maker(s) for s in hdr_iq]
            header_bytes = []
            for i in range(0, self.header_len_syms, 4):
                byte_val = (int(hdr_syms[i]) << 6) | (int(hdr_syms[i+1]) << 4) | (int(hdr_syms[i+2]) << 2) | int(hdr_syms[i+3])
                header_bytes.append(int(byte_val))

            payload_len = int(header_bytes[0])
            seq_num = int(header_bytes[1])

            pay_start = hdr_end
            total_payload_syms = (payload_len + 2) * 4
            pay_end = pay_start + total_payload_syms

            if pay_end > n_in:
                break

            passthrough_len = pre_start - in_pos

            if out_pos + passthrough_len + total_payload_syms > n_out_avail:
                break

            if passthrough_len > 0:
                out_iq[out_pos:out_pos+passthrough_len] = in_iq[in_pos:pre_start]
                out_pos += passthrough_len

            pay_iq = in_iq[pay_start:pay_end] * np.exp(-1j * phase_est) * np.conj(best_rot)
            out_iq[out_pos:out_pos+len(pay_iq)] = pay_iq

            payload_start_out_abs = self.nitems_written(0) + out_pos
            self.add_item_tag(0, payload_start_out_abs, pmt.intern("payload_len"), pmt.from_long(payload_len))
            self.add_item_tag(0, payload_start_out_abs, pmt.intern("seq_num"), pmt.from_long(seq_num))
            self.add_item_tag(0, payload_start_out_abs, pmt.intern("phase_est"), pmt.from_double(phase_est))
            self.add_item_tag(0, payload_start_out_abs, pmt.intern("ambiguity_idx"), pmt.from_long(best_rot_idx))

            out_pos += len(pay_iq)
            in_pos = pay_end

        next_tag_idx = n_in
        for t in corr_tags:
            r_idx = int(t.offset - n_read_abs)
            if r_idx >= in_pos:
                next_tag_idx = r_idx + 1
                break

        tail_passthrough = next_tag_idx - in_pos
        if tail_passthrough > 0:
            write_tail = min(tail_passthrough, n_out_avail - out_pos)
            if write_tail > 0:
                out_iq[out_pos:out_pos+write_tail] = in_iq[in_pos:in_pos+write_tail]
                out_pos += write_tail
                in_pos += write_tail

        if in_pos > 0:
            self.consume(0, in_pos)

        return out_pos

# ============================================================
# 4. Payload parser from symbol indices
# ============================================================
class payload_parser_from_symbols(gr.basic_block):
    def __init__(self):
        gr.basic_block.__init__(self, name="payload_parser_from_symbols", in_sig=[np.uint8], out_sig=None)

    def _get_tag_value_at_offset(self, tags, key_pmt, abs_offset):
        for tg in tags:
            if tg.key == key_pmt and tg.offset == abs_offset:
                return tg.value
        return None

    def general_work(self, input_items, output_items):
        syms = input_items[0]
        n = len(syms)
        if n == 0:
            return 0

        tags = list(self.get_tags_in_window(0, 0, n))

        for t in tags:
            if t.key != pmt.intern("payload_len"):
                continue

            abs_offset = t.offset
            rel_idx = int(abs_offset - self.nitems_read(0))

            payload_len_pmt = self._get_tag_value_at_offset(tags, pmt.intern("payload_len"), abs_offset)
            seq_num_pmt = self._get_tag_value_at_offset(tags, pmt.intern("seq_num"), abs_offset)

            if payload_len_pmt is None:
                continue

            payload_len = int(pmt.to_long(payload_len_pmt))
            seq_num = int(pmt.to_long(seq_num_pmt)) if seq_num_pmt is not None else -1

            pay_start = rel_idx 
            total_payload_syms = (payload_len + 2) * 4
            pay_end = pay_start + total_payload_syms

            if pay_start < 0 or pay_end > n:
                continue

            pay_syms = [int(x) for x in syms[pay_start:pay_end]]

            bytes_out = []
            for i in range(0, len(pay_syms), 4):
                b = (pay_syms[i] << 6) | (pay_syms[i+1] << 4) | (pay_syms[i+2] << 2) | (pay_syms[i+3])
                bytes_out.append(b & 0xFF)

            if len(bytes_out) < payload_len + 2:
                continue

            payload = bytes_out[:payload_len]
            crc_rx = (bytes_out[payload_len] << 8) | bytes_out[payload_len + 1]
            crc_calc = crc16_ibm(payload)

            if crc_rx == crc_calc:
                print(f"[RX Parser]  PASS | Packet #{seq_num:3d} | Len: {payload_len:2d} Bytes | Payload Head: {[hex(b) for b in payload[:4]]}...")
            else:
                print(f"[RX Parser]  FAIL | Packet #{seq_num:3d} | CRC Mismatch! RX:{hex(crc_rx)} Calc:{hex(crc_calc)}")

        self.consume(0, n)
        return 0

# ============================================================
# 5. RX Block (assemble pipeline with Equalizer & Dual AGC)
# ============================================================
class pkt_rx_QPSK(gr.hier_block2):
    def __init__(self, sps=4, samp_rate=1_000_000, alpha=0.35):
        gr.hier_block2.__init__(
            self,
            "rx_block",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(0, 0, 0)
        )

        samp_rate = samp_rate
        sym_rate = samp_rate // sps
        ntaps = 15 * sps + 1

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
            rrc
        )

        self.gain_fix = blocks.multiply_const_cc(1.0)
        self.agc1 = analog.agc2_cc(1e-3, 1e-4, 1.0, 1.0)

        # Equalizer (CMA)
        eq_algo = digital.adaptive_algorithm_cma(
            QPSK_CONST,
            0.01,
            1.0
        ).base()

        self.equalizer = digital.linear_equalizer(
            15,
            1,
            eq_algo
        )

        self.agc2 = analog.agc2_cc(1e-4, 1e-5, 1.0, 1.0)
        self.costas = digital.costas_loop_cc(0.0628, 4)

        preamble_symbols = np.array(QPSK_PREAMBLE_SYMBOLS, dtype=np.complex64)
        self.corr = digital.corr_est_cc(
            preamble_symbols.tolist(), sps=1, mark_delay=0, threshold=0.25
        )

        self.header_strip = qpsk_header_strip_with_phase(preamble_len_syms=len(QPSK_PREAMBLE_SYMBOLS), header_len_bytes=4)
        self.qpsk_decoder = digital.constellation_decoder_cb(QPSK_CONST)
        self.payload_parser_sym = payload_parser_from_symbols()

        self.qt_post = qtgui.const_sink_c(256, 'Constellation Diagram (After Costas & EQ)', 1)
        

        # Connections
        self.connect(self, self.symbol_sync)
        self.connect(self.symbol_sync, self.gain_fix)
        self.connect(self.gain_fix, self.agc1)
        self.connect(self.agc1, self.equalizer)
        self.connect(self.equalizer, self.agc2)
        self.connect(self.agc2, self.costas)

        self.connect(self.costas, self.qt_post)
        self.connect(self.costas, self.corr)
        self.connect(self.corr, self.header_strip)
        self.connect(self.header_strip, self.qpsk_decoder)
        self.connect(self.qpsk_decoder, self.payload_parser_sym)

# ============================================================
# 6. GUI Top Block (ISI Channel Added)
# ============================================================
class top_gui(Qt.QWidget):
    def __init__(self):
        super().__init__()
        self.tb = gr.top_block()

        sps = 4
        alpha = 0.35
        samp_rate = 1_000_000

        self.tx = pkt_tx_QPSK(sps, samp_rate, alpha)
        self.rx = pkt_rx_QPSK(sps, samp_rate, alpha)

        # ========================================================
        # 【新增】多路徑 ISI 通道設定 (Multipath Taps)
        # ========================================================
        # [1.0, 0.25+0.1j, 0.15-0.05j] 代表：
        # - Tap 0: 主要直射波 (Direct path)
        # - Tap 1: 第一條反射多路徑，造成前一個 Symbol 的能量重疊 (產生 ISI)
        # - Tap 2: 第二條反射多路徑 (強度較弱)
        isi_taps = [1.0 + 0.0j, 0.25 + 0.1j, 0.15 - 0.05j]

        self.channel = channels.channel_model(
            noise_voltage=0.03,        # 高斯白雜訊 (AWGN)
            frequency_offset=0.0002,   # 頻率偏差 (CFO)
            epsilon=1.0,               # 採樣率偏差 (SFO)
            taps=isi_taps,             # 【注入 ISI 通道響應】
            noise_seed=42,
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

# ============================================================
# 7. Main
# ============================================================
if __name__ == "__main__":
    qapp = Qt.QApplication(sys.argv)
    win = top_gui()
    win.setWindowTitle("QPSK: Dynamic Payload Demo with Equalizer & ISI Channel")
    win.resize(800, 600)
    win.show()
    sys.exit(qapp.exec_())