#!/usr/bin/env python3
import time
import numpy as np
import pmt
from gnuradio import blocks, digital, gr


# ==========================================
# 1. TX Packet Framer (Header + Payload 一體化打包)
# ==========================================
class PacketFramer(gr.basic_block):
    def __init__(self, payload_len=20, header_len=2, len_tag_key="payload_len"):
        gr.basic_block.__init__(
            self,
            name="PacketFramer",
            in_sig=[np.uint8],
            out_sig=[np.uint8]
        )
        self.payload_len = payload_len
        self.header_len = header_len
        self.total_len = header_len + payload_len
        self.len_tag_key = pmt.intern(len_tag_key)
        self.packet_num = 0

    def general_work(self, input_items, output_items, linker=None, item_completer=None):
        in_vec = input_items[0]
        out_vec = output_items[0]

        # 只要 Raw Payload >= 20 且 Output 空間 >= 22 就可以打包
        if (len(in_vec) >= self.payload_len) and (len(out_vec) >= self.total_len):
            # 填入 2 Bytes Header (0: 封包編號, 1: Payload 長度)
            out_vec[0] = self.packet_num & 0xFF
            out_vec[1] = self.payload_len & 0xFF

            # 填入 20 Bytes Payload
            out_vec[2 : self.total_len] = in_vec[: self.payload_len]

            # 精準打上整包總長度 (22) 的 Tag
            self.add_item_tag(
                0,
                self.nitems_written(0),
                self.len_tag_key,
                pmt.from_long(self.total_len)
            )

            self.packet_num = (self.packet_num + 1) % 256
            self.consume(0, self.payload_len)
            return self.total_len

        return 0


# ==========================================
# 2. RX Header Parser (拆解 Header 並傳送控制 Message)
# ==========================================
class CustomHeaderParser(gr.sync_block):
    def __init__(self, len_tag_key="payload_len"):
        gr.sync_block.__init__(
            self,
            name="CustomHeaderParser",
            in_sig=[np.uint8],
            out_sig=[]
        )
        self.len_tag_key = pmt.intern(len_tag_key)
        self.message_port_register_out(pmt.intern("header_data"))

    def work(self, input_items, output_items):
        in_vec = input_items[0]

        # 只要抓到 2 bytes Header 就解析
        if len(in_vec) >= 2:
            pkt_num = in_vec[0]
            p_len = in_vec[1]

            print(f"[RX Header Parser] 收到 Header -> 封包編號 (Pkt Num): {pkt_num}, 宣稱 Payload 長度: {p_len}")

            # 打包成 PMT Dictionary 回傳給 header_payload_demux
            info = pmt.make_dict()
            info = pmt.dict_add(info, self.len_tag_key, pmt.from_long(p_len))
            self.message_port_pub(pmt.intern("header_data"), info)

        return len(in_vec)


# ==========================================
# 3. RX Payload Inspector (純淨資料檢視器)
# ==========================================
class PayloadInspector(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name="PayloadInspector",
            in_sig=[np.uint8],
            out_sig=[]
        )

    def work(self, input_items, output_items):
        in_vec = input_items[0]
        if len(in_vec) >= 20:
            # 轉成純 int list 提升閱讀舒適度
            clean_data = [int(x) for x in in_vec[:20]]
            print(f"   └── [Payload Data]: {clean_data}\n")
        return len(in_vec)


# ==========================================
# 4. Top Block (乾淨流暢的 TRX Loopback)
# ==========================================
class TrxLoopbackTopBlock(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self, "TRX Clean Loopback")
        payload_len = 20
        header_len = 2
        tag_key = "payload_len"

        # --- TX 端元件 ---
        payload_data = list(range(100))
        self.src = blocks.vector_source_b(payload_data, repeat=True, vlen=1)
        self.framer = PacketFramer(
            payload_len=payload_len,
            header_len=header_len,
            len_tag_key=tag_key
        )

        # --- RX 端元件 ---
        self.rx_demux = digital.header_payload_demux(
            header_len=header_len,
            items_per_symbol=1,
            guard_interval=0,
            length_tag_key=tag_key,
            trigger_tag_key=tag_key,
            output_symbols=False,
            itemsize=gr.sizeof_char,
            timing_tag_key="",
            samp_rate=1,
            special_tags=[]
        )
        self.header_parser = CustomHeaderParser(len_tag_key=tag_key)
        self.payload_inspector = PayloadInspector()

        # ---- TX 訊號鏈路 ----
        self.connect(self.src, self.framer)
        self.connect(self.framer, self.rx_demux)

        # ---- RX 解包與監控鏈路 ----
        self.connect((self.rx_demux, 0), self.header_parser)       # Port 0: Header
        self.connect((self.rx_demux, 1), self.payload_inspector)   # Port 1: Payload

        # Message 控制鏈路 (Parser -> Demux)
        self.msg_connect((self.header_parser, "header_data"), (self.rx_demux, "header_data"))


if __name__ == "__main__":
    tb = TrxLoopbackTopBlock()
    print("=== 開始執行 簡化版 TRX Loopback ===")
    tb.start()
    time.sleep(0.2)
    tb.stop()
    tb.wait()
    time.sleep(0.1)
    print("=== 測試結束 ===")