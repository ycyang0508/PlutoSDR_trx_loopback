from gnuradio import gr, blocks, digital, filter as grfilter
from gnuradio.filter import firdes
import numpy as np
import time
import pmt

# ============================================================
#  Packet Parser (With Auto Phase-Ambiguity Correction)
# ============================================================
class packet_parser(gr.basic_block):
    def __init__(self, preamble_bits):
        gr.basic_block.__init__(
            self,
            name="packet_parser",
            in_sig=[np.complex64],
            out_sig=None
        )
        self.preamble = np.array(preamble_bits, dtype=np.uint8)
        self.p_len = len(self.preamble)
        self.buffer = []
        self.state = "SEARCH"

    def qpsk_symbol_to_bits(self, sym):
        r, i = sym.real, sym.imag
        if r > 0 and i > 0:     return [0, 0]
        elif r <= 0 and i > 0:  return [0, 1]
        elif r > 0 and i <= 0:  return [1, 0]
        else:                   return [1, 1]

    def general_work(self, input_items, output_items):
        syms = input_items[0]
        n = len(syms)
        if n <= 0:
            return 0

        self.buffer.extend(syms)

        while True:
            if self.state == "SEARCH":
                if len(self.buffer) < self.p_len:
                    break
                bits = np.array([(1 if s.real > 0 else 0) for s in self.buffer[:self.p_len]])
                if np.array_equal(bits, self.preamble):
                    self.state = "DECODE_HEADER"
                    self.buffer = self.buffer[self.p_len:]
                else:
                    self.buffer.pop(0)

            elif self.state == "DECODE_HEADER":
                needed_syms = 16
                if len(self.buffer) < needed_syms:
                    break
                header_syms = self.buffer[:needed_syms]
                self.buffer = self.buffer[needed_syms:]

                best_header = None
                for phase_idx, rot in enumerate([1, 1j, -1, -1j]):
                    bits = []
                    for s in header_syms:
                        bits.extend(self.qpsk_symbol_to_bits(s * rot))
                    header_bytes = []
                    for i in range(0, 32, 8):
                        val = 0
                        for j in range(8):
                            val = (val << 1) | bits[i+j]
                        header_bytes.append(val)
                    if header_bytes[0] == 0x10:
                        best_header = header_bytes
                        break
                    if rot == 1:
                        best_header = header_bytes

                print(f"[Packet Parser] Preamble Matched!")
                print(f"[Header Info] phase_rotation = {phase_idx}")
                print(f"[Packet Parser] Header = {[hex(b) for b in best_header]}")
                print(f"[Packet Parser] Payload length = {best_header[0]} bytes\n")
                self.state = "SEARCH"

        self.consume(0, n)
        return 0


# ============================================================
#  Decision-Directed Phase Recovery (QPSK)
# ============================================================
class phase_recovery_qpsk(gr.basic_block):
    def __init__(self, mu=0.01):
        gr.basic_block.__init__(
            self,
            name="phase_recovery_qpsk",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self.mu = mu
        self.theta = 0.0  # current phase estimate

    def slicer_qpsk(self, s):
        # QPSK slicer to nearest ideal point (unit circle)
        r = 1.0 / np.sqrt(2.0)
        re = r if s.real >= 0 else -r
        im = r if s.imag >= 0 else -r
        return re + 1j*im

    def general_work(self, input_items, output_items):
        x = input_items[0]
        y = output_items[0]
        n = min(len(x), len(y))
        if n <= 0:
            return 0

        for i in range(n):
            # rotate by current phase estimate
            z = x[i] * np.exp(-1j * self.theta)

            # decision-directed: slice to nearest QPSK point
            d = self.slicer_qpsk(z)

            # phase error = angle between z and decision d
            e = np.angle(d * np.conj(z))

            # update phase estimate
            self.theta += self.mu * e

            # output corrected symbol
            y[i] = z

        self.consume(0, n)
        return n


# ============================================================
#  TX Block (BPSK Preamble + QPSK Header)
# ============================================================
class tx_block(gr.hier_block2):
    def __init__(self, sps=4, alpha=0.35):
        gr.hier_block2.__init__(
            self,
            "tx_block",
            gr.io_signature(0,0,0),
            gr.io_signature(1,1,gr.sizeof_gr_complex)
        )

        preamble_bits = [1,0,0,1] + ([1,0]*12) + [1,0,0,1]
        header_bytes  = [0x10, 0xAA, 0xBB, 0xCC]
        payload_bytes = list(range(0x01, 0x11))

        def bytes_to_bits(bl):
            out=[]
            for b in bl:
                for i in range(8):
                    out.append((b>>(7-i))&1)
            return out

        header_bits  = bytes_to_bits(header_bytes)
        header_syms  = [(header_bits[i]<<1)|header_bits[i+1] for i in range(0,len(header_bits),2)]
        payload_bits = bytes_to_bits(payload_bytes)

        self.src_pre = blocks.vector_source_b(preamble_bits, repeat=True)
        self.src_hdr = blocks.vector_source_b(header_syms, repeat=True)
        self.src_pay = blocks.vector_source_b(payload_bits, repeat=True)

        scale = 1.0 / np.sqrt(2.0)
        qpsk_table = [(1+1j)*scale, (-1+1j)*scale, (1-1j)*scale, (-1-1j)*scale]

        self.pre_map = digital.chunks_to_symbols_bc([-1+0j, +1+0j], 1)
        self.hdr_map = digital.chunks_to_symbols_bc(qpsk_table, 1)
        self.pay_map = digital.chunks_to_symbols_bc([-1+0j, +1+0j], 1)

        self.mux = blocks.stream_mux(
            gr.sizeof_gr_complex,
            [len(preamble_bits), len(header_syms), len(payload_bits)]
        )

        rrc_taps = firdes.root_raised_cosine(1.0, sps, 1.0, alpha, 11*sps)
        self.rrc = grfilter.interp_fir_filter_ccf(sps, rrc_taps)

        self.connect(self.src_pre, self.pre_map)
        self.connect(self.src_hdr, self.hdr_map)
        self.connect(self.src_pay, self.pay_map)
        self.connect((self.pre_map,0),(self.mux,0))
        self.connect((self.hdr_map,0),(self.mux,1))
        self.connect((self.pay_map,0),(self.mux,2))
        self.connect(self.mux, self.rrc)
        self.connect(self.rrc, self)


# ============================================================
#  Channel Block
# ============================================================
class channel_block(gr.hier_block2):
    def __init__(self):
        gr.hier_block2.__init__(
            self,
            "channel_block",
            gr.io_signature(1,1,gr.sizeof_gr_complex),
            gr.io_signature(1,1,gr.sizeof_gr_complex)
        )
        self.through = blocks.copy(gr.sizeof_gr_complex)
        self.connect(self, self.through, self)


# ============================================================
#  RX Block (加上 phase_recovery_qpsk)
# ============================================================
class rx_block(gr.hier_block2):
    def __init__(self, sps=4, alpha=0.35):
        gr.hier_block2.__init__(
            self,
            "rx_block",
            gr.io_signature(1,1,gr.sizeof_gr_complex),
            gr.io_signature(0,0,0)
        )

        self.throttle = blocks.throttle(gr.sizeof_gr_complex, 200000, True)
        rrc_taps = firdes.root_raised_cosine(1.0, sps, 1.0, alpha, 11*sps)
        self.mf = grfilter.fir_filter_ccf(1, rrc_taps)
        self.timing = digital.clock_recovery_mm_cc(
            float(sps),
            0.25 * 0.175 * 0.175,
            0.5,
            0.175,
            0.005
        )
        self.costas = digital.costas_loop_cc(0.0628, 4)

        # 新增：QPSK phase recovery block
        self.phase_rec = phase_recovery_qpsk(mu=0.01)

        preamble_bits = [1,0,0,1] + ([1,0]*12) + [1,0,0,1]
        self.parser = packet_parser(preamble_bits)

        self.connect(self, self.throttle)
        self.connect(self.throttle, self.mf)
        self.connect(self.mf, self.timing)
        self.connect(self.timing, self.costas)
        self.connect(self.costas, self.phase_rec)
        self.connect(self.phase_rec, self.parser)


# ============================================================
#  Main
# ============================================================
if __name__ == "__main__":
    sps = 4
    alpha = 0.35

    tb = gr.top_block()
    tx = tx_block(sps, alpha)
    ch = channel_block()
    rx = rx_block(sps, alpha)

    tb.connect(tx, ch)
    tb.connect(ch, rx)

    tb.start()
    time.sleep(1.0)
    tb.stop()
    tb.wait()
