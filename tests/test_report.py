import json

import numpy as np
import pytest

import dsp_props as dp
from dsp_props.report import describe


def _off_by_one(x):
    return np.roll(np.fft.fft(x), 1)


@pytest.mark.full_examples
def test_failure_carries_minimal_counterexample():
    f = dp.transform(inverse=np.fft.ifft)(_off_by_one)
    report = dp.verify(f, max_examples=100, raise_on_failure=False, derandomize=True)
    inv = next(fl for fl in report.failures if fl.relation == "inversibility")
    assert isinstance(inv.example, np.ndarray)
    assert len(inv.example) <= 2  # shrunk to the smallest length where the bug shows
    assert inv.examples_tried >= 1
    assert "minimal input" in str(report)


@pytest.mark.full_examples
def test_json_and_markdown_exports():
    f = dp.transform(inverse=np.fft.ifft)(_off_by_one)
    report = dp.verify(f, max_examples=50, raise_on_failure=False, derandomize=True)
    data = json.loads(report.to_json())
    assert data["ok"] is False and data["kind"] == "transform"
    assert {r["name"] for r in data["relations"]} == set(report.checked)
    assert data["failures"][0]["counterexample"]
    md = report.to_markdown()
    assert md.startswith("### `_off_by_one` fails") and "**FAIL**" in md


def test_passing_report_has_no_failures():
    report = dp.verify(dp.transform(inverse=np.fft.ifft)(lambda x: np.fft.fft(x)), max_examples=30)
    assert report.ok and json.loads(report.to_json())["failures"] == []


def test_describe_summarises_long_signals():
    d = describe(np.zeros(100))
    assert d["N"] == 100 and "samples" not in d
    assert describe(np.array([1.0, 2.0]))["samples"] == [1.0, 2.0]
    assert describe(np.array([1 + 2j]))["samples"] == [{"re": 1.0, "im": 2.0}]
