import sys
import numpy as np
import adi
from gnuradio import gr, blocks

import numpy as np
from gnuradio import gr

class evm_generic_block(gr.sync_block):
    """
    高效能通用 EVM 計算 Block (修正版)
    支援 QPSK / 16QAM / 64QAM / 任意星座
    使用跨 Work 抽樣 + 向量化 + Circular Buffer
    """

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
        self.sample_count = 0  # 跨 work 的全局計數器

        self.buf = np.zeros(window, dtype=np.complex64)
        self.index = 0
        self.full = False

        # 參考功率（星座平均功率）
        self.ref_power = np.mean(np.abs(self.ref)**2)
        if self.ref_power == 0:
            self.ref_power = 1e-12  # 避免除以 0

        self.last_evm = 0.0

    def work(self, input_items, output_items):
        x = input_items[0]
        y = output_items[0]
        n = len(x)

        # 正確處理跨 work 的 Decimation 偏移量
        start_idx = (self.decimation - (self.sample_count % self.decimation)) % self.decimation
        self.sample_count += n

        sub_x = x[start_idx::self.decimation]

        if len(sub_x) > 0:
            # 限制單次批次量，避免記憶體與計算暴增
            if len(sub_x) > 4096:
                sub_x = sub_x[-4096:]

            # 向量化歐氏距離搜尋
            dists = np.abs(sub_x[:, None] - self.ref[None, :])
            min_indices = np.argmin(dists, axis=1)
            ref_syms = self.ref[min_indices]
            errs = sub_x - ref_syms

            # 寫入 Circular Buffer
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

            # 只要 Buffer 滿過一次，每次有新樣本進來就更新 EVM
            if self.full:
                rms_err = np.sqrt(np.mean(np.abs(self.buf)**2))
                self.last_evm = float((rms_err / np.sqrt(self.ref_power)) * 100.0)

        # 全輸出流填滿當前的 EVM 數值
        y[:] = self.last_evm
        return n

class FakePlutoSDR(gr.sync_block):
    def __init__(self, fifo_size=65536, samp_rate=1_000_000, freq_offset_hz=2000.0):
        gr.sync_block.__init__(
            self,
            name="FakePlutoSDR_FIFO_FreqOffset",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )

        self.fifo_size = fifo_size
        self.buffer = np.zeros(fifo_size, dtype=np.complex64)
        self.write_idx = 0
        self.read_idx = 0

        self.samp_rate = samp_rate
        self.freq_offset_hz = freq_offset_hz
        self.phase = 0.0
        self.phase_inc = 2 * np.pi * freq_offset_hz / samp_rate

    def work(self, input_items, output_items):
        tx = input_items[0]
        rx = output_items[0]

        n = len(tx)
        out_n = len(rx)

        # 向量化環形寫入
        for i in range(n):
            self.buffer[self.write_idx] = tx[i]
            self.write_idx = (self.write_idx + 1) % self.fifo_size

        # 向量化讀出與頻偏合成
        t = np.arange(out_n)
        phases = self.phase + self.phase_inc * t
        self.phase = (phases[-1] + self.phase_inc) % (2 * np.pi)

        # 讀取 FIFO
        for i in range(out_n):
            rx[i] = self.buffer[self.read_idx] * np.exp(1j * phases[i])
            self.read_idx = (self.read_idx + 1) % self.fifo_size

        return out_n


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
                 tx_lo=915e6, rx_lo=915e6, buf_len=16384):

        gr.hier_block2.__init__(
            self,
            "pluto_txrx_stream",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
        )

        self.buf_len = buf_len

        # Pluto SDR 初始化
        self.sdr = adi.Pluto(uri)
        self.sdr.sample_rate = int(samp_rate)

        # TX 設定
        self.sdr.tx_lo = int(tx_lo)
        self.sdr.tx_hardwaregain_chan0 = -10
        self.sdr.tx_buffer_size = buf_len
        self.sdr.tx_cyclic_buffer = False

        # RX 設定
        self.sdr.rx_lo = int(rx_lo)
        self.sdr.gain_control_mode_chan0 = 'manual'
        self.sdr.rx_hardwaregain_chan0 = 10
        self.sdr.rx_buffer_size = buf_len
        self.sdr.rx_rf_bandwidth = int(samp_rate * 2)

        # GNU Radio 內建乘法 block（TX 放大與 RX 縮小）
        self.tx_scale = blocks.multiply_const_cc(10000.0)
        self.rx_scale = blocks.multiply_const_cc(1.0 / 10000.0)

        # Pluto I/O block
        self.pluto_io = PlutoIO(buf_len, self.sdr)

        # Connect internal blocks
        self.connect(self, self.tx_scale, self.pluto_io, self.rx_scale, self)