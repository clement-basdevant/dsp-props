"""Run the internal suite on the ``fast`` cost profile when ``DSP_PROPS_PROFILE=fast`` (CI).

The tests pass explicit ``max_examples`` (an explicit count wins over the profile in
``dp.verify``); under the fast profile they are capped at the profile's count, except
for tests marked ``full_examples``, whose fault needs more inputs to be found reliably.
The scheduled CI run uses the default profile, so every test keeps its full count there.
"""

import pytest

import dsp_props as dp
import dsp_props.api


def pytest_configure(config):
    config.addinivalue_line("markers", "full_examples: keep the test's own max_examples under the fast profile")


@pytest.fixture(autouse=True)
def _cap_examples_on_fast_profile(request, monkeypatch):
    profile = dp.get_profile()
    if profile.name != "fast" or request.node.get_closest_marker("full_examples"):
        return
    real = dsp_props.api.verify

    def capped(func, *, max_examples=None, **kwargs):
        n = profile.max_examples if max_examples is None else min(max_examples, profile.max_examples)
        return real(func, max_examples=n, **kwargs)

    monkeypatch.setattr(dp, "verify", capped)
