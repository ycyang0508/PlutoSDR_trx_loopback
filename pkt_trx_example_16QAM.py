#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16QAM Continuous Packet Receiver with Header Strip & Phase Ambiguity Resolution
Updated Architecture: Clean DSP Chain Processing
"""
import sys
import time
import numpy as np
import pmt
from PyQt5 import Qt
import sip

from gnuradio import analog, blocks, digital, filter, gr, qtgui, channels
from gnuradio.filter import firdes

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
        header_bytes = [0x10, self.seq_num, self.payload_len, 0xAB]
        crc_val = crc16_ibm(payload_bytes)
        crc_bytes = [(crc_val >> 8) & 0xFF, crc_val & 0xFF]
        
        all_bytes = header_bytes + payload_bytes + crc_bytes
        
        nibbles = []
        for b in all_bytes:
            nibbles.extend([(b >> 4) & 0x0F, b & 0x0F])

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
        self.rrc = filter.interp_fir_filter_ccf(sps, rrc)
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)

        self.connect(self.pkt_gen, self.rrc, self.throttle, self)

# ============================================================
# 4. Header Strip & Ambiguity Resolution Block
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
        self.header_len_syms = self.header_len_bytes * 2
        self.qam16_const = QAM16_CONST
        self.ref_preamble = np.array(QAM16_PREAMBLE_SYMBOLS, dtype=np.complex64)
        self._rots = PHASE_ROTATIONS

        self.max_payload_bytes = int(max_payload_bytes)
        self.max_payload_syms = (self.max_payload_bytes + 2) * 2
        self.max_packet_samples = 1 + self.pre_len_syms + self.header_len_syms + self.max_payload_syms
        self.set_output_multiple(self.max_packet_samples)

        self.last_processed_offset = -1000

    def forecast(self, noutput_items, ninputs):
        return [self.max_packet_samples] * ninputs

    def _resolve_ambiguity(self, pre_iq):
        metric = [np.real(np.sum(pre_iq * np.conj(self.ref_preamble * rot))) for rot in self._rots]
        best_rot_idx = int(np.argmax(metric))
        return best_rot_idx, self._rots[best_rot_idx]

    def _estimate_cfo_and_phase(self, pre_iq):
        """利用 Preamble 計算每個 Symbol 的相位偏差並做線性擬合 (y = a*x + b)"""
        # 計算接收 Preamble 與參考 Preamble 的相位差
        phase_diff = np.angle(pre_iq * np.conj(self.ref_preamble))
        phase_unwrap = np.unwrap(phase_diff)

        # 線性擬合：a 為每 Symbol 的相位旋轉量 (CFO)，b 為初始相位
        x = np.arange(len(pre_iq))
        cfo_per_sym, phase_init = np.polyfit(x, phase_unwrap, 1)
        return cfo_per_sym, phase_init

    def general_work(self, input_items, output_items):
        # ... (前面 Tag 搜尋與長度檢查保持不變) ...

        for t in corr_tags:
            # ...
            pre_iq = in_iq[pre_start:pre_end]

            # A. 計算該幀的 CFO (a) 與 初始相位 (b)
            cfo_per_sym, phase_init = self._estimate_cfo_and_phase(pre_iq)

            # B. 針對 Header 區段進行動態相位與 CFO 補償
            hdr_raw = in_iq[hdr_start:hdr_end]
            t_hdr = np.arange(self.pre_len_syms, self.pre_len_syms + self.header_len_syms)
            hdr_iq = hdr_raw * np.exp(-1j * (cfo_per_sym * t_hdr + phase_init))

            hdr_syms = [self.qam16_const.decision_maker(s) for s in hdr_iq]
            
            header_bytes = []
            for i in range(0, self.header_len_syms, 2):
                byte_val = ((int(hdr_syms[i]) & 0x0F) << 4) | (int(hdr_syms[i+1]) & 0x0F)
                header_bytes.append(int(byte_val))

            # Header Validation
            if header_bytes[0] != 0x10 or header_bytes[3] != 0xAB:
                continue

            # ... (Payload 邊界檢查保持不變) ...

            # C. 針對 Payload 區段套用相同的 CFO 與相位補償
            pay_raw = in_iq[pay_start:pay_end]
            t_pay = np.arange(
                self.pre_len_syms + self.header_len_syms, 
                self.pre_len_syms + self.header_len_syms + total_payload_syms
            )
            pay_iq = pay_raw * np.exp(-1j * (cfo_per_sym * t_pay + phase_init))

            out_iq[out_pos:out_pos+len(pay_iq)] = pay_iq
            # ... (後續 Tag 新增與 pos 更新保持不變) ...
    def general_work(self, input_items, output_items):
        in_iq = input_items[0]
        out_iq = output_items[0]
        
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
        print(f"tag num {len(tags)}")
        for t in corr_tags:
            # 抑制距離過近的 Barker 副峰 Ghost Tag
            if t.offset - self.last_processed_offset < (self.pre_len_syms + self.header_len_syms):
                continue

            rel_idx = int(t.offset - n_read_abs)

            pre_start = rel_idx + 1
            pre_end = pre_start + self.pre_len_syms
            hdr_start = pre_end
            hdr_end = hdr_start + self.header_len_syms

            if pre_start < in_pos:
                continue

            if pre_start < 0 or hdr_end > n_in:
                break

            pre_iq = in_iq[pre_start:pre_end]
            best_rot_idx, best_rot = self._resolve_ambiguity(pre_iq)

            # 解旋轉 Header
            hdr_iq = in_iq[hdr_start:hdr_end] * np.conj(best_rot)
            hdr_syms = [self.qam16_const.decision_maker(s) for s in hdr_iq]
            
            header_bytes = []
            for i in range(0, self.header_len_syms, 2):
                byte_val = ((int(hdr_syms[i]) & 0x0F) << 4) | (int(hdr_syms[i+1]) & 0x0F)
                header_bytes.append(int(byte_val))

            # Header 格式合規檢驗: [0x10, seq, payload_len, 0xAB]
            if header_bytes[0] != 0x10 or header_bytes[3] != 0xAB:
                continue

            seq_num = int(header_bytes[1])
            payload_len = int(header_bytes[2])

            # 邊界防護：防止異常長度造成內部 Buffer Stall
            if payload_len > self.max_payload_bytes or payload_len == 0:
                continue

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

            pay_iq = in_iq[pay_start:pay_end] * np.conj(best_rot)
            out_iq[out_pos:out_pos+len(pay_iq)] = pay_iq

            payload_start_out_abs = self.nitems_written(0) + out_pos
            self.add_item_tag(0, payload_start_out_abs, pmt.intern("payload_len"), pmt.from_long(payload_len))
            self.add_item_tag(0, payload_start_out_abs, pmt.intern("seq_num"), pmt.from_long(seq_num))

            self.last_processed_offset = t.offset
            out_pos += len(pay_iq)
            in_pos = pay_end

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
# 6. Rx Block (Complete & Optimized DSP Chain)
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
        self.const = QAM16_CONST

        # 1. Matched Filter (RRC Filter)
        rrc_taps = filter.firdes.root_raised_cosine(
            gain=1.0,
            sampling_freq=samp_rate,
            symbol_rate=sym_rate,
            alpha=alpha,
            ntaps=ntaps,
        )
        self.rrc_rx = filter.fir_filter_ccf(1, rrc_taps)

        # 2. AGC (標竿參考功率設為 1.0)
        self.agc = analog.agc2_cc(1e-3, 1e-4, 1.0, 1.0)

        # 3. Symbol Timing Sync (Gardner TED)
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

        # 4. Carrier Frequency & Phase Tracking (Costas Loop)
        # loop_bw 設為 0.008，足夠穩穩定鎖定 CFO 且不跳動
        self.costas = digital.costas_loop_cc(
            loop_bw=0.008,
            order=4,
            use_snr=False
        )

        # 5. Correlation Estimator (過濾邊界調高至 0.85 壓制副峰)
        preamble_symbols = np.array(QAM16_PREAMBLE_SYMBOLS, dtype=np.complex64)
        self.corr = digital.corr_est_cc(
            preamble_symbols.tolist(),
            sps=1,
            mark_delay=0,
            threshold=0.85
        )

        # 6. Header Strip & Ambiguity Resolver
        self.header_strip = qam16_header_strip_with_phase(
            preamble_len_syms=len(QAM16_PREAMBLE_SYMBOLS),
            header_len_bytes=4,
            max_payload_bytes=256
        )

        # 7. Payload Demodulator
        self.demod = qam16_payload_demod()

        # GUI Sink
        self.copy = blocks.copy(gr.sizeof_gr_complex)
        self.qt_post = qtgui.const_sink_c(512, '16QAM Constellation', 1)

        # DSP Chain 連線
        self.connect(
            self,
            self.rrc_rx,
            self.agc,
            self.clock_sync,
            self.costas,
            self.corr,
            self.header_strip,
            self.demod
        )
        self.connect(self.header_strip, self.copy, self.qt_post)

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

        isi_taps = [1.0 + 0.0j]

        self.channel = channels.channel_model(
            noise_voltage=0.01,        # AWGN 雜訊
            frequency_offset=0.0004,   # 頻率偏差 (CFO)
            epsilon=1.0,
            taps=isi_taps,             # ISI 響應
            noise_seed=42,
            block_tags=False
        )

        self.tb.connect(self.tx, self.channel)
        self.tb.connect(self.channel, self.rx)

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