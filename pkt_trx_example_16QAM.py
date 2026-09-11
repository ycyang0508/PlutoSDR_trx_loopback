#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16QAM Continuous Packet Receiver with Header Strip & Phase Ambiguity Resolution
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
# CRC-16 (IBM)
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
QAM16_CONST = digital.constellation_16qam().base()
QAM16_POINTS = QAM16_CONST.points()

BARKER_13_RAW = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
BARKER_26_BITS = BARKER_13_RAW * 2

def barker_to_16qam_symbols(barker_list):
    return [QAM16_POINTS[0] if v == 1 else QAM16_POINTS[15] for v in barker_list]

QAM16_PREAMBLE_SYMBOLS = barker_to_16qam_symbols(BARKER_26_BITS)

# 相位旋轉因子 (0, 90, 180, 270 度)
PHASE_ROTATIONS = [1+0j, 0+1j, -1+0j, 0-1j]

# ============================================================
# 2. Sequential Packet Generator
# ============================================================
class sequential_packet_gen(gr.sync_block):
    def __init__(self, payload_len=8):
        gr.sync_block.__init__(
            self,
            name="sequential_packet_gen",
            in_sig=None,
            out_sig=[np.complex64]
        )
        self.payload_len = payload_len
        self.seq_num = 0
        self.buffer = np.array([], dtype=np.complex64)

        self.const = digital.constellation_16qam().base()
        raw_points = self.const.points()
        self.reordered_points = [None] * 16
        for pt in raw_points:
            idx = self.const.decision_maker(pt)
            self.reordered_points[idx] = pt

        self.dummy_syms = barker_to_16qam_symbols([1, -1] * 4)
        self.preamble_syms = QAM16_PREAMBLE_SYMBOLS
        self.zeros_syms = [0+0j] * 8

    def _nibbles_to_symbols(self, nibble_list):
        return [self.reordered_points[n & 0x0F] for n in nibble_list]

    def _generate_next_packet_symbols(self):
        payload_bytes = np.arange(0, self.payload_len, 1, dtype=np.uint8).tolist()
        
        # Header 格式: [0x10, Seq_Num, Payload_Len, 0xAB]
        header_bytes = [0x10, self.seq_num, self.payload_len, 0xAB]
        crc_val = crc16_ibm(payload_bytes)
        crc_bytes = [(crc_val >> 8) & 0xFF, crc_val & 0xFF]
        
        all_bytes = header_bytes + payload_bytes + crc_bytes
        
        nibbles = []
        for b in all_bytes:
            msb = (b >> 4) & 0x0F
            lsb = b & 0x0F
            nibbles.extend([msb, lsb])

        data_syms = self._nibbles_to_symbols(nibbles)

        frame_syms = np.concatenate((
            self.dummy_syms,
            self.preamble_syms,
            data_syms,
            self.zeros_syms
        )).astype(np.complex64)

        self.seq_num = (self.seq_num + 1) % 256
        return frame_syms

    def work(self, input_items, output_items):
        out = output_items[0]
        n_out = len(out)

        while len(self.buffer) < n_out:
            new_frame = self._generate_next_packet_symbols()
            self.buffer = np.concatenate((self.buffer, new_frame))

        out[:] = self.buffer[:n_out]
        self.buffer = self.buffer[n_out:]
        return n_out

# ============================================================
# 3. Tx Block
# ============================================================
class tx_block(gr.hier_block2):
    def __init__(self, sps=4, samp_rate=1_000_000, alpha=0.35):
        gr.hier_block2.__init__(
            self,
            "tx_block",
            gr.io_signature(0, 0, 0),
            gr.io_signature(1, 1, gr.sizeof_gr_complex)
        )

        sym_rate = samp_rate // sps
        ntaps = 15 * sps + 1

        self.pkt_gen = sequential_packet_gen(payload_len=8)
        rrc = firdes.root_raised_cosine(1.0, samp_rate, sym_rate, alpha, ntaps)
        self.rrc = grfilter.interp_fir_filter_ccf(sps, rrc)
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)

        self.connect(self.pkt_gen, self.rrc, self.throttle, self)

# ============================================================
# 4. 16QAM Header Strip with Phase Ambiguity Resolution Block
# ============================================================
class qam16_header_strip_with_phase(gr.basic_block):
    def __init__(self, preamble_len_syms, header_len_bytes=4, max_payload_bytes=256):
        gr.basic_block.__init__(
            self,
            name="qam16_header_strip_with_phase",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self.pre_len_syms = int(preamble_len_syms)
        self.header_len_bytes = int(header_len_bytes)
        # 16QAM: 1 Byte = 2 Symbols
        self.header_len_syms = self.header_len_bytes * 2
        self.qam16_const = QAM16_CONST
        self.ref_preamble = np.array(QAM16_PREAMBLE_SYMBOLS, dtype=np.complex64)
        self._rots = PHASE_ROTATIONS

        self.max_payload_bytes = int(max_payload_bytes)
        self.max_payload_syms = (self.max_payload_bytes + 2) * 2
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
        #print(f"[Header Strip] Found {len(corr_tags)} correlation tags in window.")
        in_pos = 0
        out_pos = 0        
        for t in corr_tags:
            rel_idx = int(t.offset - n_read_abs)

            if rel_idx < in_pos:
                continue

            # mark_delay 設為 preamble_len 時，tag 剛好指在 Preamble 後第一個位置
            pre_start = rel_idx - self.pre_len_syms
            pre_end = rel_idx
            hdr_start = pre_end
            hdr_end = hdr_start + self.header_len_syms

            if pre_start < 0 or hdr_end > n_in:
                break

            phase_est = self._get_phase_est_from_tag(t)
            pre_iq = in_iq[pre_start:pre_end] * np.exp(-1j * phase_est)
            best_rot_idx, best_rot = self._resolve_ambiguity(pre_iq)

            hdr_iq = in_iq[hdr_start:hdr_end] * np.exp(-1j * phase_est) * np.conj(best_rot)
            hdr_syms = [self.qam16_const.decision_maker(s) for s in hdr_iq]
            
            header_bytes = []
            for i in range(0, self.header_len_syms, 2):
                byte_val = ((int(hdr_syms[i]) & 0x0F) << 4) | (int(hdr_syms[i+1]) & 0x0F)
                header_bytes.append(int(byte_val))

            print([hex(b) for b in header_bytes])
            # Header Validation: [0x10, seq, payload_len, 0xAB]
            if header_bytes[0] != 0x10 or header_bytes[3] != 0xAB:
                continue

            seq_num = int(header_bytes[1])
            payload_len = int(header_bytes[2])

            pay_start = hdr_end
            total_payload_syms = (payload_len + 2) * 2
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
# 5. Payload Demodulator & CRC Checker
# ============================================================
class qam16_payload_demod(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name="qam16_payload_demod",
            in_sig=[np.complex64],
            out_sig=None
        )
        self.qam16_const = QAM16_CONST
        self.last_seq = -1

    def work(self, input_items, output_items):
        in_iq = input_items[0]
        n_in = len(in_iq)
        if n_in == 0:
            return 0

        tags = self.get_tags_in_window(0, 0, n_in)
        n_read_abs = self.nitems_read(0)

        for t in tags:
            if t.key == pmt.intern("payload_len"):
                rel_idx = int(t.offset - n_read_abs)
                payload_len = pmt.to_long(t.value)

                # 找同位置的 seq_num tag
                seq_num = -1
                for t_seq in tags:
                    if t_seq.key == pmt.intern("seq_num") and t_seq.offset == t.offset:
                        seq_num = pmt.to_long(t_seq.value)

                total_syms = (payload_len + 2) * 2
                if rel_idx + total_syms > n_in:
                    continue

                pay_iq = in_iq[rel_idx:rel_idx + total_syms]
                pay_syms = [self.qam16_const.decision_maker(s) for s in pay_iq]

                bytes_out = []
                for i in range(0, len(pay_syms) - 1, 2):
                    bytes_out.append(((pay_syms[i] & 0x0F) << 4) | (pay_syms[i+1] & 0x0F))

                payload = bytes_out[:payload_len]
                crc_rx = (bytes_out[payload_len] << 8) | bytes_out[payload_len + 1]
                crc_calc = crc16_ibm(payload)

                if crc_rx == crc_calc:
                    drop_str = ""
                    if self.last_seq != -1 and seq_num != (self.last_seq + 1) % 256:
                        missing = (seq_num - self.last_seq - 1) % 256
                        drop_str = f" [MISSING {missing} PKTS]"
                    self.last_seq = seq_num

                    print(f"[RX Demod] PASS | Packet #{seq_num:3d} | Data: {payload}{drop_str}")

        return n_in

# ============================================================
# 6. Rx Block
# ============================================================
class rx_block(gr.hier_block2):
    def __init__(self, sps=4, samp_rate=1_000_000, alpha=0.35):
        gr.hier_block2.__init__(
            self,
            "rx_block",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(0, 0, 0)
        )

        sym_rate = samp_rate // sps
        ntaps = 15 * sps + 1
        rrc = firdes.root_raised_cosine(1, samp_rate, sym_rate, alpha, ntaps)

        # 1. Symbol Sync (Gardner TED)
        self.symbol_sync = digital.symbol_sync_cc(
            digital.TED_GARDNER, sps, 0.01, 1.0, 1.0, 1.5, 1,
            QAM16_CONST, digital.IR_MMSE_8TAP, 128, rrc
        )

        # 2. AGC
        self.agc = analog.agc2_cc(1e-2, 1e-3, 1.0, 1.0)

        # 3. Costas Loop
        self.costas = digital.costas_loop_cc(loop_bw=0.01, order=4, use_snr=False)

        # 4. Correlation Estimator
        preamble_symbols = np.array(QAM16_PREAMBLE_SYMBOLS, dtype=np.complex64)
        self.corr = digital.corr_est_cc(
            preamble_symbols.tolist(),
            sps=1,
            mark_delay=len(QAM16_PREAMBLE_SYMBOLS),
            threshold=0.8
        )

        # 5. Header Strip & Ambiguity Resolver
        self.header_strip = qam16_header_strip_with_phase(
            preamble_len_syms=len(QAM16_PREAMBLE_SYMBOLS),
            header_len_bytes=4,
            max_payload_bytes=256
        )

        # 6. Payload Demodulator
        self.demod = qam16_payload_demod()

        self.qt_post = qtgui.const_sink_c(512, '16QAM Constellation', 1)

        self.connect(self, self.symbol_sync)
        self.connect(self.symbol_sync, self.agc)
        self.connect(self.agc, self.costas)
        self.connect(self.costas, self.corr)
        self.connect(self.corr, self.header_strip)
        self.connect(self.header_strip, self.demod)
        self.connect(self.costas, self.qt_post)

# ============================================================
# 7. GUI Top Block
# ============================================================
class top_gui(Qt.QWidget):
    def __init__(self):
        super().__init__()
        self.tb = gr.top_block()

        sps = 4
        alpha = 0.35
        samp_rate = 1_000_000

        self.tx = tx_block(sps, samp_rate, alpha)
        self.rx = rx_block(sps, samp_rate, alpha)

        #self.channel = channels.channel_model(
        #    noise_voltage=0.005,
        #    frequency_offset=0.0000,
        #    epsilon=1.0,
        #    taps=[1.0 + 0.0j],
        #    noise_seed=42,
        #    block_tags=False
        #)

        #self.tb.connect(self.tx, self.channel)
        #self.tb.connect(self.channel, self.rx)
        self.tb.connect(self.tx, self.rx)

        layout = Qt.QVBoxLayout()
        self.setLayout(layout)
        self.layout().addWidget(sip.wrapinstance(self.rx.qt_post.qwidget(), Qt.QWidget))

        self.tb.start()

    def closeEvent(self, event):
        self.tb.stop()
        self.tb.wait()
        event.accept()

if __name__ == "__main__":
    qapp = Qt.QApplication(sys.argv)
    win = top_gui()
    win.setWindowTitle("16QAM Header Strip Receiver with Phase Ambiguity")
    win.resize(800, 600)
    win.show()
    sys.exit(qapp.exec_())