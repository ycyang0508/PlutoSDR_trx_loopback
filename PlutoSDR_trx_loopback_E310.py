#!/usr/bin/env python3
import sys
import numpy as np
import adi
from PyQt5 import Qt
import sip
from gnuradio import gr, blocks, analog, digital, qtgui, filter, fft
from custom_blocks import *
from collections import deque

# ---------------------------------------------------------
#  EVM Block (具備自動 Scaling 與 Rotations 判斷)
# ---------------------------------------------------------
class evm_generic_block(gr.sync_block):
    def __init__(self, constellation_points, window=2048, skip_samples=8192):
        gr.sync_block.__init__(
            self,
            name="evm_generic",
            in_sig=[np.complex64],
            out_sig=[np.float32],
        )
        self.ref = np.array(constellation_points, dtype=np.complex64)
        self.window = window
        self.skip_samples = skip_samples
        self.processed = 0
        self.buf = np.zeros(window, dtype=np.complex64)
        self.index = 0
        self.full = False
        self.ref_power = np.mean(np.abs(self.ref)**2) or 1e-12
        self.last_evm = 0.0

        self.rotations = [1.0, 1j, -1.0, -1j]

    def work(self, input_items, output_items):
        x = input_items[0]
        y = output_items[0]
        n = len(x)

        if self.processed < self.skip_samples:
            take = min(n, self.skip_samples - self.processed)
            self.processed += take
            x = x[take:]
            if len(x) == 0:
                y[:] = self.last_evm
                return n

        sub_x = x
        if len(sub_x) > 0:
            if len(sub_x) > 4096:
                sub_x = sub_x[-4096:]

            best_errs = None
            min_total_err = float('inf')

            for rot in self.rotations:
                rot_x = sub_x * rot
                
                dists = np.abs(rot_x[:, None] - self.ref[None, :])
                min_idx = np.argmin(dists, axis=1)
                ref_syms = self.ref[min_idx]

                scale = np.real(np.sum(rot_x * np.conj(ref_syms))) / (np.sum(np.abs(rot_x)**2) + 1e-12)
                scaled_x = rot_x * scale
                errs = scaled_x - ref_syms
                
                total_err = np.sum(np.abs(errs)**2)
                if total_err < min_total_err:
                    min_total_err = total_err
                    best_errs = errs

            n_errs = len(best_errs)
            if n_errs >= self.window:
                self.buf[:] = best_errs[-self.window:]
                self.index = 0
                self.full = True
            else:
                space = self.window - self.index
                if n_errs <= space:
                    self.buf[self.index:self.index + n_errs] = best_errs
                    self.index += n_errs
                else:
                    self.buf[self.index:] = best_errs[:space]
                    self.buf[:n_errs - space] = best_errs[space:]
                    self.index = n_errs - space
                    self.full = True

            if self.full:
                rms_err = np.sqrt(np.mean(np.abs(self.buf)**2))
                self.last_evm = float((rms_err / np.sqrt(self.ref_power)) * 100.0)

        y[:] = self.last_evm
        return n

# ---------------------------------------------------------
#  PlutoSDR TX/RX Block
# ---------------------------------------------------------
class PlutoIO(gr.sync_block):
    def __init__(self, buf_len, sdr):
        gr.sync_block.__init__(
            self,
            name="pluto_io",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self.buf_len = buf_len
        self.sdr = sdr
        self.set_output_multiple(buf_len)
        self.sdr.rx_buffer_size = buf_len

        self.rx_buf = np.array([], dtype=np.complex64)

        for _ in range(5):
            _ = self.sdr.rx()

    def work(self, input_items, output_items):
        in_data = input_items[0]
        out_data = output_items[0]
        n_in  = len(in_data)
        n_out = len(out_data)

       #for i in range(0, n_in, self.buf_len):
       #    end = min(i + self.buf_len, n_in)
       #    tx_chunk = in_data[i:end]
       #    if len(tx_chunk) > 0:
       #        self.sdr.tx(tx_chunk)
        self.sdr.tx(in_data)

        while len(self.rx_buf) < n_out:
            rx_chunk = self.sdr.rx()
            self.rx_buf = np.concatenate([self.rx_buf, rx_chunk])

        out_data[:] = self.rx_buf[:n_out]
        self.rx_buf = self.rx_buf[n_out:]

        return n_out

class PlutoSDR_txrx_stream(gr.hier_block2):
    def __init__(self, uri="ip:192.168.1.10", samp_rate=1_000_000,
                 tx_lo=915e6, rx_lo=915e6, buf_len=32768):

        gr.hier_block2.__init__(
            self,
            "pluto_txrx_stream",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
        )

        self.sdr = adi.Pluto(uri)
        self.sdr.sample_rate = int(samp_rate)

        try:
            self.sdr.quadrature_tracking_en = True
            self.sdr.rfdc_tracking_en = True
            self.sdr.bbdc_tracking_en = False
        except AttributeError:
            pass

        self.sdr.tx_lo = int(tx_lo)
        self.sdr.tx_hardwaregain_chan0 = 0
        self.sdr.tx_buffer_size = buf_len
        self.sdr.tx_cyclic_buffer = False

        self.sdr.rx_lo = int(rx_lo)
        self.sdr.gain_control_mode_chan0 = 'manual'
        self.sdr.rx_hardwaregain_chan0 = 20
        self.sdr.rx_buffer_size = buf_len
        self.sdr.rx_rf_bandwidth = int(samp_rate * 2)

        # scaling
        self.tx_scale = blocks.multiply_const_cc(10000.0)
        self.rx_scale = blocks.multiply_const_cc(1.0 / 10000.0)

        # throttle blocks
        self.tx_throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)
        self.rx_throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)

        # Pluto IO
        self.pluto_io = PlutoIO(buf_len, self.sdr)

        # TX path: input → scale → throttle → Pluto
        self.connect(self, self.tx_throttle, self.tx_scale, self.pluto_io)

        # RX path: Pluto → throttle → scale → output
        self.connect(self.pluto_io,  self.rx_scale, self.rx_throttle,self)


# ---------------------------------------------------------
#  QPSK TX Block
# ---------------------------------------------------------
class QPSK_TX_block(gr.hier_block2):
    def __init__(self, sps=4, samp_rate=1_000_000, rolloff=0.35):
        gr.hier_block2.__init__(
            self,
            "QPSK_TX_block",
            gr.io_signature(1, 1, gr.sizeof_char),
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
        )

        sym_rate = samp_rate // sps
        ntaps = 15 * sps + 1

        const = digital.constellation_qpsk().base()
        points = const.points()

        self.mapper = digital.chunks_to_symbols_bc(points, 1)

        self.rrc_tx = filter.interp_fir_filter_ccf(
            sps,
            filter.firdes.root_raised_cosine(
                gain=1.0,
                sampling_freq=samp_rate,
                symbol_rate=sym_rate,
                alpha=rolloff,
                ntaps=ntaps
            )
        )

        self.connect(self, self.mapper, self.rrc_tx, self)

# ---------------------------------------------------------
#  QPSK RX Block（輸出 symbols + bits）
# ---------------------------------------------------------
class QPSK_RX_block(gr.hier_block2):
    def __init__(self, sps=4, samp_rate=1_000_000, rolloff=0.35):
        gr.hier_block2.__init__(
            self,
            "QPSK_RX_block",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signaturev(2, 2, [gr.sizeof_gr_complex, gr.sizeof_char]),
        )

        sym_rate = samp_rate // sps
        ntaps = 15 * sps + 1

        rrc_taps = filter.firdes.root_raised_cosine(
            gain=1.0,
            sampling_freq=samp_rate,
            symbol_rate=sym_rate,
            alpha=rolloff,
            ntaps=ntaps
        )

        self.rrc_rx = filter.fir_filter_ccf(1, rrc_taps)
        self.agc = analog.agc2_cc(1e-3, 1e-4, 1.41421356, 1.0)

        self.clock_sync = digital.symbol_sync_cc(
            digital.TED_GARDNER,
            sps,
            0.001,
            1.0,
            1.0,
            1.5,
            1,
            digital.constellation_qpsk().base(),
            digital.IR_MMSE_8TAP
        )

        self.costas = digital.costas_loop_cc(0.0003, 4, False)
        self.copy = blocks.copy(gr.sizeof_gr_complex)

        self.const = digital.constellation_qpsk().base()
        self.decoder = digital.constellation_decoder_cb(self.const)
        self.sym2bit = digital.map_bb([0, 1, 3, 2])
        self.unpack = blocks.unpack_k_bits_bb(2)

        self.connect(self, self.rrc_rx, self.agc,
                     self.clock_sync, self.costas, self.copy)

        self.connect(self.copy, (self, 0))
        self.connect(self.costas, self.decoder, self.sym2bit, self.unpack, (self, 1))

# ---------------------------------------------------------
#  主程式
# ---------------------------------------------------------
class qpsk_cable_demo(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self)

        samp_rate = 1_000_000
        sps = 4
        buf_len = 16384

        const = digital.constellation_qpsk().base()
        raw_pts = np.array(const.points(), dtype=np.complex64)
        points = raw_pts / np.abs(raw_pts[0])

        n_symbols = 200000
        rnd = np.random.randint(0, 4, n_symbols).tolist()
        self.src = blocks.vector_source_b(rnd, repeat=True)

        self.tx = QPSK_TX_block(sps=sps, samp_rate=samp_rate)
        self.rx = QPSK_RX_block(sps=sps, samp_rate=samp_rate)

        self.pluto = PlutoSDR_txrx_stream(
            uri="ip:192.168.1.10",
            samp_rate=samp_rate,
            tx_lo=915e6,
            rx_lo=915e6,
            buf_len=buf_len
        )

        self.freq_sink = qtgui.freq_sink_c(
            8192,
            fft.window.WIN_HAMMING,
            0,
            samp_rate,
            "Rx Spectrum (1MHz)"
        )
        self.freq_win = sip.wrapinstance(self.freq_sink.qwidget(), Qt.QWidget)
        self.freq_sink.set_fft_average(0.3)

        self.const_sink = qtgui.const_sink_c(
            1024,
            "Cable Loopback QPSK (1MHz)",
            1
        )
        self.const_sink.set_x_axis(-2.0, 2.0)
        self.const_sink.set_y_axis(-2.0, 2.0)
        self.const_win = sip.wrapinstance(self.const_sink.qwidget(), Qt.QWidget)


        self.evm = evm_generic_block(points, window=2048, skip_samples=32768)

        self.evm_sink = qtgui.number_sink(
            gr.sizeof_float,
            0.5,
            qtgui.NUM_GRAPH_HORIZ,
            1,
            None
        )
        self.evm_sink.set_title("EVM (%)")
        self.evm_sink.set_update_time(2.0)
        self.evm_sink.set_min(0, -20)
        self.evm_sink.set_max(0, 20)
        self.evm_sink_win = sip.wrapinstance(self.evm_sink.qwidget(), Qt.QWidget)
               

        # 新增：bit output 的空 sink
        self.bit_sink = blocks.null_sink(gr.sizeof_char)

        # TX path
        self.connect(self.src, self.tx, self.pluto)

        # Spectrum
        self.connect(self.pluto, self.freq_sink)

        # RX symbols + bits
        self.connect(self.pluto, self.rx)

        # constellation
        self.connect((self.rx, 0), (self.const_sink, 0))

        # EVM
        self.connect((self.rx, 0), self.evm)
        self.connect(self.evm, self.evm_sink)

        # 新增：bit output 接到 null sink
        self.connect((self.rx, 1), self.bit_sink)


def run_demo():
    app = Qt.QApplication(sys.argv)
    tb = qpsk_cable_demo()

    win = Qt.QWidget()
    layout = Qt.QVBoxLayout(win)
    layout.addWidget(tb.freq_win)
    layout.addWidget(tb.const_win)
    layout.addWidget(tb.evm_sink_win)
    win.show()

    tb.start()
    app.exec_()
    tb.stop()
    tb.wait()

if __name__ == "__main__":
    run_demo()
