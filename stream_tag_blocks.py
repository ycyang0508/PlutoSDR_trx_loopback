#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from gnuradio import gr, blocks
import pmt

# ---------------------------------------------------------
# Python Tagger Block (sync_block)
# ---------------------------------------------------------
class python_tagger(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name="python_tagger",
            in_sig=[np.float32],
            out_sig=[np.float32]
        )

    def work(self, input_items, output_items):
        x = input_items[0]
        y = output_items[0]
        print(f"work() called, nitems_written={self.nitems_written(0)}")
        wr_base = self.nitems_written(0)

        # 每 50 個 sample 塞一個 tag
        for i, sample in enumerate(x):
            if (wr_base + i) % 50 == 0:
                self.add_item_tag(
                    0,
                    wr_base + i,
                    pmt.intern("test_tag"),
                    pmt.from_long(wr_base + i)
                )

        y[:] = x
        return len(y)

# ---------------------------------------------------------
# Tag Collector Block
# ---------------------------------------------------------
class tag_collector(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name="tag_collector",
            in_sig=[np.float32],
            out_sig=[np.float32]
        )
        self.collected_tags = []   # ⭐ 用來收集所有 tag

    def work(self, input_items, output_items):
        x = input_items[0]
        y = output_items[0]
        nread = self.nitems_read(0)
        nin = len(x)

        # ⭐ 抓這次 work() 裡的所有 tag
        tags = self.get_tags_in_window(0, 0, nin)

        for t in tags:
            key = pmt.symbol_to_string(t.key)
            value = t.value
            offset = t.offset

            # ⭐ 存起來
            self.collected_tags.append((offset, key, value))

        # passthrough
        y[:] = x
        return len(y)

# ---------------------------------------------------------
# Top Block (Flowgraph)
# ---------------------------------------------------------
class top_block(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self)

        # 產生隨機 float 資料
        self.src = blocks.vector_source_f(
            np.random.rand(500).astype(np.float32),
            repeat=False
        )

        # 加 throttle（float）
        self.throttle = blocks.throttle(gr.sizeof_float, 10000, True)

        # Python Tagger
        self.tagger = python_tagger()

        # Tag Collector
        self.collector = tag_collector()

        # Tag Debug（可選）
        self.tag_debug = blocks.tag_debug(
            gr.sizeof_float, "Tag Debug", "test_tag"
        )

        # Null Sink
        self.snk = blocks.null_sink(gr.sizeof_float)

        # Connect
        self.connect(self.src, self.throttle)
        self.connect(self.throttle, self.tagger)
        self.connect(self.tagger, self.collector)
        self.connect(self.collector, self.tag_debug)
        self.connect(self.collector, self.snk)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    tb = top_block()
    tb.run()

    print("\n=== All Collected Tags ===")
    for offset, key, value in tb.collector.collected_tags:
        print(f"offset={offset}, key={key}, value={value}")
