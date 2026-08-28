#!/usr/bin/env python3
import sys
import numpy as np
import adi
import sip
from gnuradio import gr, blocks


# =========================================================
#  1. PlutoSDR 獨立 TX Block
# =========================================================
class PlutoTX(gr.sync_block):
    def __init__(self, uri="ip:192.168.1.10", samp_rate=1_000_000, 
                 tx_lo=915e6, tx_gain=0, buf_len=32768, sdr=None):
        gr.sync_block.__init__(
            self,
            name="pluto_tx",
            in_sig=[np.complex64],
            out_sig=[]  # TX 是 Sink Block
        )
        self.buf_len = buf_len
        self.sdr = sdr if sdr else adi.Pluto(uri)
        
        self.sdr.sample_rate = int(samp_rate)
        self.sdr.tx_lo = int(tx_lo)
        self.sdr.tx_hardwaregain_chan0 = int(tx_gain)
        self.sdr.tx_buffer_size = buf_len
        self.sdr.tx_cyclic_buffer = False

        # 初始化 TX 緩衝區 (FIFO)
        self.tx_buf = np.array([], dtype=np.complex64)

    def work(self, input_items, output_items):
        in_data = input_items[0]
        
        # 1. 將 GNU Radio 上游進來的數據 append 到內部 FIFO
        if len(in_data) > 0:
            self.tx_buf = np.concatenate([self.tx_buf, in_data])

        # 2. 只要 FIFO 內的數據大於等於一個標準硬體 Buffer，就切割並送出
        while len(self.tx_buf) >= self.buf_len:
            tx_chunk = self.tx_buf[:self.buf_len]
            self.sdr.tx(tx_chunk)                 # 永遠傳送固定長度 (buf_len)
            self.tx_buf = self.tx_buf[self.buf_len:] # 移除已發送的數據

        # 3. 告知 GNU Radio 已消耗完本輪輸入的所有採樣點
        return len(in_data)


# =========================================================
#  2. PlutoSDR 獨立 RX Block
# =========================================================
class PlutoRX(gr.sync_block):
    def __init__(self, uri="ip:192.168.1.10", samp_rate=1_000_000, 
                 rx_lo=915e6, rx_gain=20, buf_len=32768, sdr=None):
        gr.sync_block.__init__(
            self,
            name="pluto_rx",
            in_sig=[],  # RX 是 Source 節點，沒有輸入
            out_sig=[np.complex64]
        )
        self.buf_len = buf_len
        # 若外部沒傳入共享 sdr，則自行建立實例
        self.sdr = sdr if sdr else adi.Pluto(uri)

        self.sdr.sample_rate = int(samp_rate)
        self.sdr.rx_lo = int(rx_lo)
        self.sdr.gain_control_mode_chan0 = 'manual'
        self.sdr.rx_hardwaregain_chan0 = int(rx_gain)
        self.sdr.rx_buffer_size = buf_len
        self.sdr.rx_rf_bandwidth = int(samp_rate * 2)

        try:
            self.sdr.quadrature_tracking_en = True
            self.sdr.rfdc_tracking_en = True
            self.sdr.bbdc_tracking_en = False
        except AttributeError:
            pass

        self.rx_buf = np.array([], dtype=np.complex64)

        # 預先拋棄前幾幀無效資料（Flush Hardware Buffer）
        for _ in range(5):
            _ = self.sdr.rx()

    def work(self, input_items, output_items):
        out_data = output_items[0]
        n_out = len(out_data)

        # 持續拉取硬體數據，直到滿足 output_items 的長度需求
        while len(self.rx_buf) < n_out:
            rx_chunk = self.sdr.rx()
            self.rx_buf = np.concatenate([self.rx_buf, rx_chunk])

        out_data[:] = self.rx_buf[:n_out]
        self.rx_buf = self.rx_buf[n_out:]

        return n_out

# ---------------------------------------------------------
#  PlutoSDR TX/RX Block
# ---------------------------------------------------------
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
        #self.pluto_io = PlutoIO_rx(buf_len, self.sdr)
        self.sdr_tx = PlutoTX(uri="ip:192.168.1.10",samp_rate=samp_rate,tx_lo=tx_lo,buf_len=buf_len,sdr=self.sdr)
        self.sdr_rx = PlutoRX(uri="ip:192.168.1.10",samp_rate=samp_rate,rx_lo=tx_lo,buf_len=buf_len,sdr=self.sdr)

        # TX path: input → scale → throttle → Pluto
        self.connect(self, self.tx_throttle, self.tx_scale, self.sdr_tx)

        # RX path: Pluto → throttle → scale → output
        self.connect(self.sdr_rx,  self.rx_scale, self.rx_throttle,self)


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