#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full QPSK demo:
- mux TX (preamble + header(payload_len) + payload + CRC)
- channel_model
- RX: symbol_sync -> AGC -> Costas -> corr_est_cc -> header strip (remove preamble+header,
       apply coarse phase & ambiguity correction to payload) -> constellation_decoder -> payload parser (CRC check)
- GUI constellation tap (after Costas)
"""
import sys
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
# 1. Constellation
# ============================================================
QPSK_CONST = digital.constellation_qpsk().base()
QPSK_POINTS = QPSK_CONST.points()

BARKER_13_RAW = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
BARKER_26_BITS = BARKER_13_RAW * 2

def barker_to_qpsk_symbols(barker_list):
    return [QPSK_POINTS[0] if v == 1 else QPSK_POINTS[3] for v in barker_list]

QPSK_PREAMBLE_SYMBOLS = barker_to_qpsk_symbols(BARKER_26_BITS)

# ============================================================
# 2. TX Block (mux version, payload + CRC)
# ============================================================
class tx_block(gr.hier_block2):
    def __init__(self, sps=4, alpha=0.35, payload_len=16):
        gr.hier_block2.__init__(
            self,
            "tx_block",
            gr.io_signature(0,0,0),
            gr.io_signature(1,1,gr.sizeof_gr_complex)
        )

        samp_rate = 100000
        sym_rate = samp_rate // sps
        ntaps = 15 * sps + 1

        # helper: bytes -> symbol indices (0..3)
        def bytes_to_symidx(bl):
            out = []
            for b in bl:
                for shift in (6,4,2,0):
                    out.append((b >> shift) & 0x03)
            return out

        # 1) Dummy (complex symbols)
        dummy_syms = barker_to_qpsk_symbols([1, -1] * 32)
        self.src_dummy = blocks.vector_source_c(dummy_syms, repeat=True)

        # 2) Preamble (complex symbols)
        self.src_pre = blocks.vector_source_c(QPSK_PREAMBLE_SYMBOLS, repeat=True)

        # 3) Header (first byte = payload length)
        header_bytes = [payload_len, 0xAA, 0xCC, 0xEE]
        header_symidx = bytes_to_symidx(header_bytes)
        self.src_hdr = blocks.vector_source_b(header_symidx, repeat=True)
        self.map_hdr = digital.chunks_to_symbols_bc(QPSK_POINTS, 1)

        # 4) Payload + CRC
        payload_bytes = np.random.randint(0, 256, payload_len, dtype=np.uint8).tolist()
        #print(f"[TX] Payload bytes: { [hex(b) for b in payload_bytes] }")
        crc_val = crc16_ibm(payload_bytes)
        crc_bytes = [(crc_val >> 8) & 0xFF, crc_val & 0xFF]
        payload_full = payload_bytes + crc_bytes
        payload_symidx = bytes_to_symidx(payload_full)
        self.src_pay = blocks.vector_source_b(payload_symidx, repeat=True)
        self.map_pay = digital.chunks_to_symbols_bc(QPSK_POINTS, 1)

        # 5) Zero padding
        zeros_syms = [0+0j] * 64
        self.src_zero = blocks.vector_source_c(zeros_syms, repeat=True)

        # MUX lengths (in complex samples)
        len_dummy = len(dummy_syms)
        len_pre = len(QPSK_PREAMBLE_SYMBOLS)
        len_hdr = len(header_symidx)
        len_pay = len(payload_symidx)
        len_zero = len(zeros_syms)

        self.mux = blocks.stream_mux(
            gr.sizeof_gr_complex,
            [len_dummy, len_pre, len_hdr, len_pay, len_zero]
        )

        # RRC shaping
        rrc = firdes.root_raised_cosine(1.0, samp_rate, sym_rate, alpha, ntaps)
        self.rrc = grfilter.interp_fir_filter_ccf(sps, rrc)
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)

        # Connect
        self.connect(self.src_dummy, (self.mux, 0))
        self.connect(self.src_pre,   (self.mux, 1))
        self.connect(self.src_hdr, self.map_hdr, (self.mux, 2))
        self.connect(self.src_pay, self.map_pay, (self.mux, 3))
        self.connect(self.src_zero, (self.mux, 4))

        self.connect(self.mux, self.rrc, self.throttle, self)

# ============================================================
# 3. Header strip + phase correction block
#    - remove preamble + header from stream
#    - apply coarse phase & ambiguity correction to payload before output
#    - add tags at payload start in output: payload_len, phase_est, ambiguity_idx
# ============================================================

class qpsk_header_strip_with_phase(gr.basic_block):
    """
    解析 preamble+header，剝掉 preamble+header，並把 payload 輸出前做 phase + ambiguity 補償。
    修復了 forecast 轉型錯誤、防止滑動視窗尾端被誤吞導致的封包斷裂。
    """
    def __init__(self, preamble_len_syms, header_len_bytes=4, max_payload_bytes=256):
        gr.basic_block.__init__(self,
                               name="qpsk_header_strip_with_phase",
                               in_sig=[np.complex64],
                               out_sig=[np.complex64])
        self.pre_len_syms = int(preamble_len_syms)
        self.header_len_bytes = int(header_len_bytes)
        self.header_len_syms = self.header_len_bytes * 4
        self.qpsk_const = QPSK_CONST
        self.ref_preamble = np.array(QPSK_PREAMBLE_SYMBOLS, dtype=np.complex64)
        self._rots = [1.0, 1j, -1.0, -1j]

        self.max_payload_bytes = int(max_payload_bytes)
        self.max_payload_syms = (self.max_payload_bytes + 2) * 4
        # 一個完整封包最大可能需要的點數
        self.max_packet_samples = 1 + self.pre_len_syms + self.header_len_syms + self.max_payload_syms
        self.set_output_multiple(self.max_packet_samples)


    def forecast(self, noutput_items, ninputs):
        # 要求 scheduler 至少提供一個完整 packet 的 input
        need = self.max_packet_samples
        ninput_items_required = [need] * ninputs
        return ninput_items_required
       
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
        best_rot = rotations[best_rot_idx]
        return best_rot_idx, best_rot

    def general_work(self, input_items, output_items):
        in_iq = input_items[0]
        out_iq = output_items[0]

        n_in = len(in_iq)
        n_out_avail = len(out_iq)
        if n_in == 0 or n_out_avail == 0:
            return 0

        # 取得當前 window 的所有 tags
        tags = self.get_tags_in_window(0, 0, n_in)
        corr_tags = [t for t in tags if t.key == pmt.intern("corr_start")]
        corr_tags.sort(key=lambda x: int(x.offset))

        # 絕對讀取基底
        n_read_abs = self.nitems_read(0)

        # 如果沒有找到任何標籤，代表這整段都是普通噪訊或無用訊號，直接 Passthrough 輸出
        if not corr_tags:
            write_len = min(n_in, n_out_avail)
            out_iq[:write_len] = in_iq[:write_len]
            self.consume(0, write_len)
            return write_len

        # 追蹤我們處理到 input 的哪個位置 (相對 index)
        in_pos = 0
        out_pos = 0
        #print(f"[HeaderStrip] n_in={n_in}, n_out_avail={n_out_avail}, tags={len(corr_tags)}")
        for t in corr_tags:
            # 計算 tag 在當前 input_items 中的相對位置
            rel_idx = int(t.offset - n_read_abs)
            
            # 如果這個標籤的位置在我們已經處理過的 in_pos 之前，直接跳過
            if rel_idx < in_pos:
                continue

            pre_start = rel_idx + 1
            pre_end = pre_start + self.pre_len_syms
            hdr_start = pre_end
            hdr_end = hdr_start + self.header_len_syms

            # 檢查 1：如果連 Header 都拿不全，說明封包斷在 window 邊界
            # 停止處理，保留現狀，等待更多數據進來
            if hdr_end > n_in:
                break

            # 讀取相角並解析 Header 內容以獲取 Payload 長度
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

            # 計算 Payload 邊界
            pay_start = hdr_end
            total_payload_syms = (payload_len + 2) * 4
            pay_end = pay_start + total_payload_syms

            # 檢查 2：如果 Payload 點數不夠，說明後半段還沒進來
            # 停止處理，保留這個標籤之後的訊號
            if pay_end > n_in:
                break

            # 計算當前位置到封包起點之前的普通訊號長度 (Passthrough 段)
            passthrough_len = pre_start - in_pos
            
            # 檢查 3：檢查 Output 空間是否足夠容納 (Passthrough 訊號 + Payload 訊號)
            if out_pos + passthrough_len + total_payload_syms > n_out_avail:
                # 空間不足，為了避免破壞封包，我們在此中斷，把處理權交還 Scheduler
                break

            # --- 開始寫入 Output ---
            # 1) 複製封包前的 Passthrough 訊號
            if passthrough_len > 0:
                out_iq[out_pos:out_pos+passthrough_len] = in_iq[in_pos:pre_start]
                out_pos += passthrough_len

            # 2) 補償並複製 Payload 訊號
            pay_iq = in_iq[pay_start:pay_end] * np.exp(-1j * phase_est) * np.conj(best_rot)
            out_iq[out_pos:out_pos+len(pay_iq)] = pay_iq

            # 3) 附加新的標籤到 Output
            payload_start_out_abs = self.nitems_written(0) + out_pos
            self.add_item_tag(0, payload_start_out_abs, pmt.intern("payload_len"), pmt.from_long(payload_len))
            self.add_item_tag(0, payload_start_out_abs, pmt.intern("phase_est"), pmt.from_double(phase_est))
            self.add_item_tag(0, payload_start_out_abs, pmt.intern("ambiguity_idx"), pmt.from_long(best_rot_idx))

            out_pos += len(pay_iq)
            in_pos = pay_end  # 更新已消耗的 input 指標

            phase_deg = np.degrees(phase_est)
            #print(f"[HeaderStrip] Processed packet: Header={ [hex(b) for b in header_bytes] }, len={payload_len}, phase={phase_deg:.1f}°, amb={best_rot_idx}, t.offset={t.offset}")

        # 如果處理完完整的封包後，後面還殘留一些訊號，且這些訊號在所有已知標籤之前（或是已經沒標籤了）
        # 我們可以安全地進行流式 Passthrough，直到下一個「未處理的封包」或 window 邊界
        # 為了絕對安全，若有未處理完的 tag 留著，我們只 passthrough 到那個 tag 的 pre_start 之前
        next_tag_idx = n_in
        for t in corr_tags:
            r_idx = int(t.offset - n_read_abs)
            if r_idx >= in_pos:
                next_tag_idx = r_idx + 1 # 包含開頭那個點
                break

        tail_passthrough = next_tag_idx - in_pos
        if tail_passthrough > 0:
            write_tail = min(tail_passthrough, n_out_avail - out_pos)
            if write_tail > 0:
                out_iq[out_pos:out_pos+write_tail] = in_iq[in_pos:in_pos+write_tail]
                out_pos += write_tail
                in_pos += write_tail

        # 確實消耗掉已經處理完成的數據
        if in_pos > 0:
            self.consume(0, in_pos)

        return out_pos


# ============================================================
# 4. Payload parser from symbol indices (after constellation_decoder_cb)
#    - input: uint8 symbols (0..3)
#    - expects tags at payload start: payload_len (pmt long)
# ============================================================
class payload_parser_from_symbols(gr.basic_block):
    """
    Input: symbol indices stream (uint8) from constellation_decoder_cb
    Expects tags at payload start (payload_len)
    """
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
            if payload_len_pmt is None:
                continue

            payload_len = int(pmt.to_long(payload_len_pmt))

            # 修正點：Header 已經被剝離，rel_idx 本身就是 Payload 起點
            pay_start = rel_idx 
            total_payload_syms = (payload_len + 2) * 4  # Payload + 2 bytes CRC
            pay_end = pay_start + total_payload_syms

            if pay_start < 0 or pay_end > n:
                continue

            pay_syms = [int(x) for x in syms[pay_start:pay_end]]

            # 將 4 個 2-bit symbols 組合回 1 個 Byte
            bytes_out = []
            for i in range(0, len(pay_syms), 4):
                b = (pay_syms[i] << 6) | (pay_syms[i+1] << 4) | (pay_syms[i+2] << 2) | (pay_syms[i+3])
                bytes_out.append(b & 0xFF)

            if len(bytes_out) < payload_len + 2:
                continue

            payload = bytes_out[:payload_len]
            crc_rx = (bytes_out[payload_len] << 8) | bytes_out[payload_len + 1]
            crc_calc = crc16_ibm(payload)

            #print(f"[Payload Parser] payload_len={payload_len} | payload={[hex(b) for b in payload]}")
            if crc_rx == crc_calc:
                pass
                #print(f"[Payload CRC] PASS | RX CRC={hex(crc_rx)}")
            else:
                print(f"[Payload CRC] FAIL | RX CRC={hex(crc_rx)} Calc={hex(crc_calc)}")

        self.consume(0, n)
        return 0
# ============================================================
# 5. RX Block (assemble pipeline)
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
        self.agc = analog.agc2_cc(1e-3, 1e-4, 1, 1.0)
        self.costas = digital.costas_loop_cc(0.0628, 4)

        preamble_symbols = np.array(QPSK_PREAMBLE_SYMBOLS, dtype=np.complex64)
        self.corr = digital.corr_est_cc(
            preamble_symbols.tolist(), sps=1, mark_delay=0, threshold=0.30
        )

        # header strip (remove preamble+header, output corrected payload)
        self.header_strip = qpsk_header_strip_with_phase(preamble_len_syms=len(QPSK_PREAMBLE_SYMBOLS), header_len_bytes=4)

        # decoder: complex -> symbol indices (uint8)
        self.qpsk_decoder = digital.constellation_decoder_cb(QPSK_CONST)

        # payload parser (from symbol indices)
        self.payload_parser_sym = payload_parser_from_symbols()

        # GUI const sink
        self.qt_pre = qtgui.const_sink_c(256, 'Constellation Diagram', 1)

        # connections
        self.connect(self, self.symbol_sync)
        self.connect(self.symbol_sync, self.gain_fix)
        self.connect(self.gain_fix, self.agc)
        self.connect(self.agc, self.costas)

        # tap constellation at Costas
        self.connect(self.costas, self.qt_pre)

        # corr -> header_strip -> decoder -> payload parser
        self.connect(self.costas, self.corr)
        self.connect(self.corr, self.header_strip)
        self.connect(self.header_strip, self.qpsk_decoder)
        self.connect(self.qpsk_decoder, self.payload_parser_sym)

# ============================================================
# 6. GUI Top Block
# ============================================================
class top_gui(Qt.QWidget):
    def __init__(self):
        super().__init__()
        self.tb = gr.top_block()

        sps = 4
        alpha = 0.35
        payload_len = 16

        self.tx = tx_block(sps, alpha, payload_len=payload_len)
        self.rx = rx_block(sps, alpha)

        self.channel = channels.channel_model(
            noise_voltage=0.05,
            frequency_offset=0.0002,
            epsilon=1.0,
            taps=[1.0+0j],
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
    win.setWindowTitle("QPSK: header strip + payload CRC demo")
    win.resize(800, 600)
    win.show()
    sys.exit(qapp.exec_())
