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

#BARKER_13_RAW = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
#BARKER_26_BITS = BARKER_13_RAW * 2

def barker_to_16qam_symbols(barker_list):
    return [QAM16_POINTS[2] if v == 1 else QAM16_POINTS[8] for v in barker_list]

#QAM16_PREAMBLE_SYMBOLS = barker_to_16qam_symbols(BARKER_26_BITS)
#preamble = QAM16_PREAMBLE_SYMBOLS

preamble = np.array([
    0.316+0.316j,
    0.948+0.316j,
    0.316+0.948j,
    -0.316+0.316j,
    -0.948+0.316j,
    -0.316+0.948j,
    0.948+0.948j,
    -0.948+0.948j,      

], dtype=np.complex64)

QAM16_PREAMBLE_SYMBOLS = preamble



RX_SEARCH_PREAMBLE = 0
RX_HEADER_PHASE    = 1
RX_PAYLOAD         = 2

HEADER_BYTES    = 4
HEADER_SYMBOLS  = HEADER_BYTES*2

PAYLOAD_BYTES    = 8
PAYLOAD_SYMBOL   = PAYLOAD_BYTES*2


PRE_LEN  = len(QAM16_PREAMBLE_SYMBOLS)   # 你的 preamble 長度


PHASE_ROTATIONS = [1+0j, 0+1j, -1+0j, 0-1j]

# ============================================================
# 2. Sequential Packet Generator
# ============================================================
class sequential_packet_gen(gr.sync_block):
    def __init__(self, min_payload_len=4, max_payload_len=32):
        gr.sync_block.__init__(
            self,
            name="sequential_packet_gen",
            in_sig=None,
            out_sig=[np.complex64]
        )
        self.min_payload_len = min_payload_len
        self.max_payload_len = max_payload_len
        #self.payload_len = payload_len
        self.seq_num = 0
        self.buffer = np.array([], dtype=np.complex64)

        self.const = digital.constellation_16qam().base()
        raw_points = self.const.points()
        self.reordered_points = [None] * 16
        for pt in raw_points:
            idx = self.const.decision_maker(pt)
            self.reordered_points[idx] = pt

        self.dummy_syms = barker_to_16qam_symbols([1, -1] * 16)
        self.preamble_syms = QAM16_PREAMBLE_SYMBOLS
        self.zeros_syms = [0+0j] * 8

    def _nibbles_to_symbols(self, nibble_list):
        return [self.reordered_points[n & 0x0F] for n in nibble_list]

    def _generate_next_packet_symbols(self):
        #payload_bytes = np.arange(0, self.payload_len, 1, dtype=np.uint8).tolist()
        self.payload_len = np.random.randint(self.min_payload_len, self.max_payload_len + 1)
        payload_bytes = np.arange(0, self.payload_len, 1, dtype=np.uint8).tolist()
               
        header_bytes = [0x10, self.seq_num, self.payload_len, 0x00]
        chk_sum = np.uint8((header_bytes[0] + header_bytes[1] + header_bytes[2]) & 0xFF)
        header_bytes[3] = chk_sum
        #print(f"tx header {[hex(b) for b in header_bytes]}")

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

        nwrite = self.nitems_written(0)

        while len(self.buffer) < n_out:
            new_frame = self._generate_next_packet_symbols()

            # preamble 起始位置（在 new_frame 裡）
            pre_start = len(self.dummy_syms)  # dummy_syms 之後就是 preamble
            # 在 TX 端加 tag：preamble_start
            self.add_item_tag(
                                0,
                                nwrite + len(self.buffer) + pre_start,
                                pmt.intern("preamble_start"),
                                pmt.from_long(self.seq_num)
                            )

            self.buffer = np.concatenate((self.buffer, new_frame))

        out[:] = self.buffer[:n_out]
        self.buffer = self.buffer[n_out:]
        return n_out

# ============================================================
# 3. Tx Block
# ============================================================
class pkt_tx_16QAM(gr.hier_block2):
    def __init__(self, sps=4, samp_rate=1_000_000, alpha=0.35):
        gr.hier_block2.__init__(
            self,
            "tx_block",
            gr.io_signature(0, 0, 0),
            gr.io_signature(1, 1, gr.sizeof_gr_complex)
        )

        sym_rate = samp_rate // sps
        ntaps = 15 * sps + 1

        self.pkt_gen = sequential_packet_gen(min_payload_len=PAYLOAD_BYTES,max_payload_len=PAYLOAD_BYTES*2)
        rrc = firdes.root_raised_cosine(sps, samp_rate, sym_rate, alpha, ntaps)
        self.rrc = filter.interp_fir_filter_ccf(sps, rrc)
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)

        self.connect(self.pkt_gen, self.rrc, self.throttle, self)

class preamble_detector_cc(gr.sync_block):
    def __init__(self, preamble_syms):
        gr.sync_block.__init__(
            self,
            name="preamble_detector_cc",
            in_sig=[np.complex64],
            out_sig=[np.complex64],
        )

        self.ref = np.array(preamble_syms, dtype=np.complex64)
        self.pre_len = len(self.ref)
        self.candidates = np.array([0, np.pi/2, np.pi, 3*np.pi/2], dtype=np.float32)

        self.history = np.array([], dtype=np.complex64)
        self.last_mag = 0.0
        self.threshold = 0.95
        self.ref_norm = np.linalg.norm(self.ref)

    def _resolve_phase(self, rx_pre):
        # 向量化相位候選比較
        # rx_pre shape: (pre_len,), candidates shape: (4, 1)
        rotated = rx_pre * np.exp(-1j * self.candidates[:, None])
        dists = np.sum(np.abs(rotated - self.ref)**2, axis=1)
        best_idx = np.argmin(dists)
        return self.candidates[best_idx], dists[best_idx]

    def work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]
        n = len(inp)
        if n == 0:
            return 0

        nwrite = self.nitems_written(0)

        # 1. 結合歷史資料與當前輸入
        full_data = np.concatenate([self.history, inp])

        # 2. 向量化計算 Normalized Correlation (取代原本的 for 迴圈)
        # 分子：使用 np.correlate 計算滑動內積
        numerators = np.correlate(full_data, self.ref, mode='valid')

        # 分母：利用累加和 (cumsum) 快速計算每個滑動視窗的能量平方和
        sq_data = np.abs(full_data)**2
        cumsum_sq = np.concatenate(([0.0], np.cumsum(sq_data)))
        window_sums = cumsum_sq[self.pre_len:] - cumsum_sq[:-self.pre_len]
        denominators = np.sqrt(np.maximum(window_sums, 0.0)) * self.ref_norm + 1e-12

        mags = np.abs(numerators) / denominators

        # 3. 找出所有高於門檻值的候選點索引
        candidate_indices = np.where(mags >= self.threshold)[0]

        # 4. 僅對通過門檻的點檢查 Local Maximum 並解算相位
        for i in candidate_indices:
            left_val = self.last_mag if i == 0 else mags[i - 1]
            right_val = 0.0 if i == n - 1 else mags[i + 1]

            if mags[i] >= left_val and mags[i] >= right_val:
                window = full_data[i : i + self.pre_len]
                best_phi, _ = self._resolve_phase(window)
                pre_start = nwrite + i 

                self.add_item_tag(
                    0,
                    pre_start,
                    pmt.intern("preamble_match"),                            
                    pmt.from_double(best_phi)
                )

        # 5. 更新歷史記錄與邊界狀態
        if n >= self.pre_len - 1:
            self.history = inp[-(self.pre_len - 1):]
        else:
            self.history = full_data[-(self.pre_len - 1):]
        
        self.last_mag = mags[-1] if len(mags) > 0 else 0.0

        out[:] = inp
        return n

class packet_parsing(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name="packet_parsing",
            in_sig=[np.complex64],
            out_sig=None
        )

        # 定義狀態常數
        self.RX_SEARCH_PREAMBLE = 0
        self.RX_HEADER_PHASE     = 1
        self.RX_PAYLOAD          = 2

        self.state = self.RX_SEARCH_PREAMBLE
        self.phase = 0.0
        
        # 收集暫存與計數
        self.header_syms_needed = HEADER_BYTES * 2  # 每個 Byte 佔 2 個 16QAM Symbols (Nibbles)
        self.payload_syms_needed = 0
        self.current_seq = 0
        self.current_payload_len = 0
        self.collected_syms = []

        self.qam16_const = QAM16_CONST

    def work(self, input_items, output_items):
        inp = input_items[0]        
        n = len(inp)
        nread = self.nitems_read(0)

        # 檢索當前 Window 內的 Tags
        tags = self.get_tags_in_window(0, 0, n)
        rx_tags_dict = {int(t.offset - nread): t for t in tags if t.key == pmt.intern("preamble_match")}

        for i in range(n):
            s = inp[i]
            abs_idx = nread + i

            # ----------------------------------------------------
            # 狀態 1：搜尋 Preamble
            # ----------------------------------------------------
            if self.state == self.RX_SEARCH_PREAMBLE:
                if i in rx_tags_dict:
                    tag = rx_tags_dict[i]
                    self.phase = pmt.to_python(tag.value)
                    
                    # 驗證 Preamble 對齊（可選）
                    start = i - PRE_LEN
                    if start >= 0:
                        preamble_rx = inp[start:i] * np.exp(-1j * self.phase)
                        #print(f"\n[RX State] Preamble Matched at idx={i}, phase={self.phase:.4f}")
                        #print(f"preamble_rx ={preamble_rx}")
                        #print(f"preamble_ref={preamble}")

                    # 進入 Header 收集狀態
                    self.state = self.RX_HEADER_PHASE
                    self.collected_syms = []
                continue

            # 對後續所有訊號進行即時相位修正
            s_corrected = s * np.exp(-1j * self.phase)

            # ----------------------------------------------------
            # 狀態 2：收集與解析 Header (4 Bytes = 8 Symbols)
            # ----------------------------------------------------
            if self.state == self.RX_HEADER_PHASE:
                self.collected_syms.append(s_corrected)
                
                if len(self.collected_syms) == self.header_syms_needed:
                    # 進行星座圖判決 (Decision Maker)
                    hard_syms = [self.qam16_const.decision_maker(sym) for sym in self.collected_syms]
                    
                    # 將 Nibbles 組裝回 Bytes: [0x10, seq, payload_len, checksum]
                    header_bytes = []
                    for k in range(0, len(hard_syms), 2):
                        b_val = ((int(hard_syms[k]) & 0x0F) << 4) | (int(hard_syms[k+1]) & 0x0F)
                        header_bytes.append(int(b_val))
                    #print(f"header_bytes {[hex(_b) for _b in header_bytes]}")
                    # 檢查 Checksum
                    chk_sum = np.uint8((header_bytes[0] + header_bytes[1] + header_bytes[2]) & 0xFF)
                    if header_bytes[3] == chk_sum:
                        self.current_seq = header_bytes[1]
                        self.current_payload_len = header_bytes[2]
                        
                        print(f"[RX Header] PASS | Seq: {self.current_seq}, Payload Len: {self.current_payload_len}")
                        
                        # 準備進入 Payload 階段 (Payload + 2 Bytes CRC)
                        self.state = self.RX_PAYLOAD
                        self.payload_syms_needed = (self.current_payload_len + 2) * 2
                        self.collected_syms = []
                    else:
                        print(f"[RX Header] FAIL | Checksum mismatch: {hex(header_bytes[3])} != {hex(chk_sum)}")
                        # 失敗則重置回搜尋狀態
                        self.state = self.RX_SEARCH_PREAMBLE
                        self.collected_syms = []

            # ----------------------------------------------------
            # 狀態 3：收集 Payload 與 CRC 檢查
            # ----------------------------------------------------
            elif self.state == self.RX_PAYLOAD:
                self.collected_syms.append(s_corrected)

                if len(self.collected_syms) == self.payload_syms_needed:
                    hard_syms = [self.qam16_const.decision_maker(sym) for sym in self.collected_syms]

                    bytes_out = []
                    for k in range(0, len(hard_syms) - 1, 2):
                        bytes_out.append(((hard_syms[k] & 0x0F) << 4) | (hard_syms[k+1] & 0x0F))

                    payload = bytes_out[:self.current_payload_len]
                    crc_rx = (bytes_out[self.current_payload_len] << 8) | bytes_out[self.current_payload_len + 1]
                    crc_calc = crc16_ibm(payload)

                    if crc_rx == crc_calc:                                                
                        print(f"[RX Payload] SUCCESS 🎉 | Seq #{self.current_seq} | Data: {payload}")
                    else:
                        print(f"[RX Payload] CRC ERROR ❌ | Rx: {hex(crc_rx)} vs Calc: {hex(crc_calc)} | Data: {payload}")

                    # 處理完一個封包後，回到初始狀態繼續尋找下一個 Preamble
                    self.state = self.RX_SEARCH_PREAMBLE
                    self.collected_syms = []

        self.consume(0, n)            
        return 0


# ============================================================
# 5. Payload Demodulator & CRC Checker
# ============================================================

# ============================================================
# 6. Rx Block (Complete & Optimized DSP Chain)
# ============================================================
class pkt_rx_16QAM(gr.hier_block2):
    def __init__(self, sps=4, samp_rate=1_000_000, alpha=0.35,eq_taps=15, eq_gain=0.001):
        gr.hier_block2.__init__(
            self,
            "rx_block",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(0, 0, 0)
        )

        

        sym_rate = samp_rate // sps
        ntaps = 15 * sps + 1
        self.const = QAM16_CONST

        self.throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)

        # 1. Matched Filter (RRC Filter)
        rrc_taps = filter.firdes.root_raised_cosine(
            gain=sps,
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
            detector_type = digital.TED_GARDNER          ,
            sps           = sps,
            loop_bw       = 0.001,
            damping_factor= 1.0,
            ted_gain      = 1.0,
            max_deviation = 1.5,
            osps          = 1,
            slicer        = self.const,
            interp_type   = digital.IR_MMSE_8TAP,
            n_filters     = ntaps,
            taps          = rrc_taps
        )
            
        
        self.eq_alg = digital.adaptive_algorithm_cma(self.const, eq_gain,1.0)
        self.eq = digital.linear_equalizer(
            num_taps=eq_taps,
            sps=1,  # Clock Sync 輸出已降至 1 sps
            alg=self.eq_alg,
            adapt_after_training=False
        )
        
        # 4. Carrier Frequency & Phase Tracking (Costas Loop)
        # loop_bw 設為 0.008，足夠穩穩定鎖定 CFO 且不跳動        
        self.costas = digital.costas_loop_cc(
            loop_bw=0.008,
            order=4,
            use_snr=False
        )
        

        self.preamble_det = preamble_detector_cc(QAM16_PREAMBLE_SYMBOLS)
        
        
        #self.throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate/sps, True)
        
        

        self.pkt_parsing = packet_parsing()

        # 7. Payload Demodulator
        #self.demod = qam16_payload_demod()

        # GUI Sink
        self.copy = blocks.copy(gr.sizeof_gr_complex)
        self.qt_post = qtgui.const_sink_c(1024, '16QAM Constellation', 1)
        self.null_sink = blocks.null_sink(gr.sizeof_gr_complex)

        # DSP Chain 連線
        self.connect(
                     self,         
                     self.throttle,
                     self.rrc_rx,
                     self.agc,
                     self.clock_sync,      # 先鎖 timing                                             
                     self.eq,                     
                     self.costas,                       
                     self.preamble_det,    # 在 Costas 之前做 preamble + phase 解旋
                     #self.null_sink
                     self.pkt_parsing,
                    )
        

        self.connect(self.preamble_det, self.qt_post)

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

        self.tx = pkt_tx_16QAM(sps, samp_rate, alpha)
        self.rx = pkt_rx_16QAM(sps, samp_rate, alpha)
        
        isi_taps = [1.0 + 0.0j, 0.25 + 0.1j, 0.15 - 0.05j]

        self.channel = channels.channel_model(
            noise_voltage=0.004,        # AWGN 雜訊
            frequency_offset=0.002,   # 頻率偏差 (CFO)
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
