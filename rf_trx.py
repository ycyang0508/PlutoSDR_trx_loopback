#!/usr/bin/env python3
import sys
import numpy as np
import adi
import sip
from gnuradio import gr, blocks




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


# -------------------------------------------------------------------
# 模擬 PlutoSDR 的 ZeroMQ Stream Hier Block (純軟體迴路)
# -------------------------------------------------------------------
class PlutoSDR_zmq_txrx_stream(gr.hier_block2):

    def __init__(
        self,
        uri="ip:192.168.1.10",  # 保留介面相容性（內部不使用）
        samp_rate=1_000_000,
        tx_lo=915e6,
        rx_lo=915e6,
        buf_len=32768,
        zmq_rx_addr="tcp://127.0.0.1:5555",
        zmq_tx_addr="tcp://127.0.0.1:5556",
    ):

        gr.hier_block2.__init__(
            self,
            "pluto_zmq_txrx_stream",
            gr.io_signature(0, 0, 0),
            gr.io_signature(0, 0, 0),
        )

        # 1. 實例化內部 Buffer Loopback 核心
        self.loopback_core = InternalBufferLoopback(buf_len=buf_len)

        # 2. Rate Control (Throttle 確保讀寫速度符合 sample_rate)
        self.throttle = blocks.throttle(
            gr.sizeof_gr_complex, samp_rate, True
        )

        # 3. ZeroMQ 端點配置
        # TX: SUB Source 接收 Python PUB (bind=False)
        self.zmq_tx_source = zeromq.pull_source(gr.sizeof_gr_complex, 1, zmq_tx_addr, 1000, False, -1, False )

        # RX: PUSH Sink 傳送給 Python PULL (bind=True)
        self.zmq_rx_sink = zeromq.push_sink(
            gr.sizeof_gr_complex, 1, zmq_rx_addr, 1000, False, -1, True
        )

        # 4. 拓撲連線：ZMQ TX Source -> Loopback Buffer -> Throttle -> ZMQ RX Sink
        self.connect(self.zmq_tx_source, self.loopback_core)
        self.connect(self.loopback_core, self.throttle, self.zmq_rx_sink)