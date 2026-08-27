#!/usr/bin/env python3
import sys
import numpy as np
import adi
import sip
from custom_blocks import *

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