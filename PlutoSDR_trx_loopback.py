#!/usr/bin/env python3
import sys
import numpy as np
import adi
from PyQt5 import Qt
import sip

from gnuradio import gr, blocks, analog, digital, qtgui, filter, fft


# ---------------------------------------------------------
#  EVM Block
# ---------------------------------------------------------
class evm_generic_block(gr.sync_block):
    def __init__(self, constellation_points, window=2048, decimation=16):
        gr.sync_block.__init__(
            self,
            name="evm_generic",
            in_sig=[np.complex64],
            out_sig=[np.float32],
        )

        self.ref = np.array(constellation_points, dtype=np.complex64)
        self.window = window
        self.decimation = max(1, decimation)
        self.sample_count = 0
        self.buf = np.zeros(window, dtype=np.complex64)
        self.index = 0
        self.full = False
        self.ref_power = np.mean(np.abs(self.ref)**2) or 1e-12
        self.last_evm = 0.0

    def work(self, input_items, output_items):
        x = input_items[0]
        y = output_items[0]
        n = len(x)

        start_idx = (self.decimation - (self.sample_count % self.decimation)) % self.decimation
        self.sample_count += n
        sub_x = x[start_idx::self.decimation]

        if len(sub_x) > 0:
            if len(sub_x) > 4096:
                sub_x = sub_x[-4096:]

            dists = np.abs(sub_x[:, None] - self.ref[None, :])
            min_indices = np.argmin(dists, axis=1)
            ref_syms = self.ref[min_indices]
            errs = sub_x - ref_syms

            n_errs = len(errs)
            if n_errs >= self.window:
                self.buf[:] = errs[-self.window:]
                self.index = 0
                self.full = True
            else:
                space = self.window - self.index
                if n_errs <= space:
                    self.buf[self.index:self.index + n_errs] = errs
                    self.index += n_errs
                else:
                    self.buf[self.index:] = errs[:space]
                    self.buf[:n_errs - space] = errs[space:]
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

    def work(self, input_items, output_items):
        in_data = input_items[0]
        out_data = output_items[0]
        n_out = len(out_data)

        for i in range(0, n_out, self.buf_len):
            end = min(i + self.buf_len, n_out)

            tx_chunk = in_data[i:end]
            if len(tx_chunk) > 0:
                self.sdr.tx(tx_chunk)

            rx_chunk = self.sdr.rx()
            out_data[i:end] = rx_chunk[:(end - i)]

        return n_out


class PlutoSDR_txrx_stream(gr.hier_block2):
    def __init__(self, uri="ip:192.168.2.1", samp_rate=1_000_000,
                 tx_lo=915e6, rx_lo=915e6, buf_len=32768):

        gr.hier_block2.__init__(
            self,
            "pluto_txrx_stream",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
        )

        self.sdr = adi.Pluto(uri)
        self.sdr.sample_rate = int(samp_rate)

        self.sdr.tx_lo = int(tx_lo)
        self.sdr.tx_hardwaregain_chan0 = -10
        self.sdr.tx_buffer_size = buf_len
        self.sdr.tx_cyclic_buffer = False

        self.sdr.rx_lo = int(rx_lo)
        self.sdr.gain_control_mode_chan0 = 'manual'
        self.sdr.rx_hardwaregain_chan0 = 10
        self.sdr.rx_buffer_size = buf_len
        self.sdr.rx_rf_bandwidth = int(samp_rate * 2)

        self.tx_scale = blocks.multiply_const_cc(10000.0)
        self.rx_scale = blocks.multiply_const_cc(1.0 / 10000.0)

        self.pluto_io = PlutoIO(buf_len, self.sdr)

        self.connect(self, self.tx_scale, self.pluto_io, self.rx_scale, self)


# ---------------------------------------------------------
#  QPSK TX Block（修正 input 連接）
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
        ntaps = 11 * sps + 1

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

        # 修正：外部輸入 → mapper → RRC → output
        self.connect(self, self.mapper, self.rrc_tx, self)


# ---------------------------------------------------------
#  QPSK RX Block
# ---------------------------------------------------------
class QPSK_RX_block(gr.hier_block2):
    def __init__(self, sps=4, samp_rate=1_000_000, rolloff=0.35):
        gr.hier_block2.__init__(
            self,
            "QPSK_RX_block",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
        )

        sym_rate = samp_rate // sps
        ntaps = 11 * sps + 1

        rrc_taps = filter.firdes.root_raised_cosine(
            gain=1.0,
            sampling_freq=samp_rate,
            symbol_rate=sym_rate,
            alpha=rolloff,
            ntaps=ntaps
        )

        self.dc_block = filter.dc_blocker_cc(128, True)
        self.agc = analog.agc2_cc(1e-2, 1e-3, 1.0, 1.0)
        self.rrc_rx = filter.fir_filter_ccf(1, rrc_taps)

        self.clock_sync = digital.symbol_sync_cc(
            digital.TED_GARDNER,
            sps,
            0.015,
            1.0,
            1.0,
            1.5,
            1,
            digital.constellation_qpsk().base(),
            digital.IR_MMSE_8TAP
        )

        self.costas = digital.costas_loop_cc(0.015, 4, False)
        self.copy = blocks.copy(gr.sizeof_gr_complex)

        self.connect(self, self.dc_block, self.agc, self.rrc_rx,
                     self.clock_sync, self.costas, self.copy)

        self.connect(self.copy, self)


# ---------------------------------------------------------
#  主程式
# ---------------------------------------------------------
class qpsk_cable_demo(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self)

        samp_rate = 1_000_000
        sps = 4
        buf_len = 32768

        const = digital.constellation_qpsk().base()
        points = const.points()

        # TX bit source
        n_symbols = 200000
        rnd = np.random.randint(0, 4, n_symbols).tolist()
        self.src = blocks.vector_source_b(rnd, repeat=True)

        # TX / RX blocks
        self.tx = QPSK_TX_block(sps=sps, samp_rate=samp_rate)
        self.rx = QPSK_RX_block(sps=sps, samp_rate=samp_rate)

        # Pluto SDR
        self.pluto = PlutoSDR_txrx_stream(
            uri="ip:192.168.2.1",
            samp_rate=samp_rate,
            tx_lo=915e6,
            rx_lo=915e6,
            buf_len=buf_len
        )

        # Spectrum
        self.freq_sink = qtgui.freq_sink_c(
            2048,
            fft.window.WIN_HAMMING,
            0,
            samp_rate,
            "Rx Spectrum (1MHz)"
        )
        self.freq_win = sip.wrapinstance(self.freq_sink.qwidget(), Qt.QWidget)

        # Constellation
        self.const_sink = qtgui.const_sink_c(
            4096,
            "Cable Loopback QPSK (1MHz)",
            1
        )
        self.const_sink.set_x_axis(-2.0, 2.0)
        self.const_sink.set_y_axis(-2.0, 2.0)
        self.const_win = sip.wrapinstance(self.const_sink.qwidget(), Qt.QWidget)

        # EVM
        self.evm_sink = qtgui.number_sink(
            gr.sizeof_float,
            0.5,
            qtgui.NUM_GRAPH_HORIZ,
            1,
            None
        )
        self.evm_sink.set_title("EVM (%)")
        self.evm_sink.set_update_time(4.0)
        self.evm_sink_win = sip.wrapinstance(self.evm_sink.qwidget(), Qt.QWidget)

        self.evm = evm_generic_block(points, window=2048, decimation=16)

        # Connections
        self.connect(self.src, self.tx, self.pluto)
        self.connect(self.pluto, self.freq_sink)
        self.connect(self.pluto, self.rx)
        self.connect(self.rx, self.const_sink)
        self.connect(self.rx, self.evm, self.evm_sink)


# ---------------------------------------------------------
#  run_demo()
# ---------------------------------------------------------
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
