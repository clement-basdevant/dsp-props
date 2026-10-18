import numpy as np

import dsp_props as dp


class MovingSum:
    """Correct 3-tap moving sum with carried history."""

    def __init__(self):
        self.hist = np.zeros(2)

    def process(self, block):
        buf = np.concatenate([self.hist, block])
        out = buf[2:] + buf[1:-1] + buf[:-2]
        self.hist = buf[-2:]
        return out


class ForgetfulMovingSum(MovingSum):
    """Mutant: history is reset at every block."""

    def process(self, block):
        self.hist = np.zeros(2)
        return super().process(block)


class ForgetsAtTenthCall(MovingSum):
    """Mutant: history is lost at the tenth call only (counter or ring buffer overflow)."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def process(self, block):
        self.calls += 1
        if self.calls == 10:
            self.hist = np.zeros(2)
        return super().process(block)


def test_correct_block_processor_passes():
    assert dp.verify(dp.streaming()(MovingSum), max_examples=100).ok


def test_state_reset_mutant_is_detected():
    report = dp.verify(dp.streaming()(ForgetfulMovingSum), max_examples=100, raise_on_failure=False)
    assert not report.ok


def test_state_lost_after_many_blocks_is_detected():
    report = dp.verify(dp.streaming()(ForgetsAtTenthCall), max_examples=100, raise_on_failure=False)
    assert not report.ok
    x, cuts = report.failures[0].example
    assert len(cuts) + 1 >= 10


def test_stateful_wrapper():
    def accumulate(block, total):
        out = total + np.cumsum(block)
        return out, (out[-1] if len(out) else total)

    assert dp.verify(dp.stateful(accumulate, 0.0), max_examples=100).ok
