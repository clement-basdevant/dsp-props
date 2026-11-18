import numpy as np
import pytest
import scipy.signal as ss

import dsp_props as dp
from dsp_props.core import IIRTolerance, longdouble_lfilter, norm, recursion_gain


def test_recursion_gain_one_pole():
    # 1 / (1 - 0.5 z^-1): h[n] = 0.5^n, ||h||_1 = 2
    assert np.isclose(recursion_gain([1.0, -0.5]), 2.0, rtol=1e-9)


def test_unstable_filter_has_infinite_gain():
    assert recursion_gain([1.0, -1.5]) == np.inf


def test_bound_holds_for_resonant_filters():
    rng = np.random.default_rng(1)
    for r in (0.9, 0.99, 0.999):
        poles = r * np.exp(1j * np.array([0.4, -0.4, 1.1, -1.1]))
        a = np.real(np.poly(poles))
        b = rng.uniform(-1, 1, 5)
        x = rng.standard_normal(128)
        err = norm(ss.lfilter(b, a, x) - longdouble_lfilter(b, a, x).astype(float))
        assert err <= IIRTolerance(b, a).bound(128, np.float64, norm(x))


def test_lti_filter_api_on_scipy_lfilter():
    b, a = [0.2, 0.3], [1.0, -0.9]
    f = dp.lti_filter(tolerance=IIRTolerance(b, a))(lambda x: ss.lfilter(b, a, x))
    assert dp.verify(f, max_examples=100).ok


@pytest.mark.full_examples
def test_non_causal_filter_declared_causal_is_caught():
    f = dp.lti_filter(causal=True)(lambda x: np.convolve(x, [0.25, 0.5, 0.25], mode="same"))
    report = dp.verify(f, max_examples=100, raise_on_failure=False)
    assert "causality" in [fl.relation for fl in report.failures]


@pytest.mark.parametrize("a", [[1.0, -1.5], [1.0, -1.0], [1.0, 0.0, 1.0], [2.0, -2.5]])
def test_unstable_transfer_is_rejected(a):
    # poles at 1.5, 1 (integrator), +-i and 1.25: the error bound would be infinite
    b = [1.0]
    with pytest.raises(ValueError, match="unstable"):
        dp.lti_filter(transfer=(b, a))(lambda x: ss.lfilter(b, a, x))
    with pytest.raises(ValueError, match="unstable"):
        dp.lti_filter(transfer=(b, a), tolerance=dp.core.FFTTolerance(c=1000.0))(lambda x: x)
    with pytest.raises(ValueError, match="unstable"):
        IIRTolerance(b, a)


def test_stable_and_fir_transfers_are_accepted():
    dp.lti_filter(transfer=([0.2, 0.3], [1.0, -0.999]))(lambda x: x)
    dp.lti_filter(transfer=([0.25, 0.5, 0.25], [1.0]))(lambda x: x)


# Poles close to the unit circle: the measured impulse response is capped at
# 2^16 samples, long before it decays (r^65536 = 1.4e-3 for r = 0.9999).
NEAR_UNIT_POLE = {
    "dc blocker r=0.9999": ([1.0, -1.0], [1.0, -0.9999]),
    "dc blocker r=0.99999": ([1.0, -1.0], [1.0, -0.99999]),
    "butter(2, 5e-5)": tuple(ss.butter(2, 5e-5)),
}


@pytest.mark.parametrize("name", list(NEAR_UNIT_POLE))
def test_correct_filter_with_pole_near_unit_circle_passes(name):
    b, a = NEAR_UNIT_POLE[name]
    f = dp.lti_filter(transfer=(b, a))(lambda x: ss.lfilter(b, a, x))
    report = dp.verify(f, max_examples=50, raise_on_failure=False)
    assert report.ok, report


@pytest.mark.full_examples
@pytest.mark.parametrize("wrong", ["pole", "gain"])
def test_wrong_filter_with_pole_near_unit_circle_is_caught(wrong):
    b, a = [1.0, -1.0], [1.0, -0.9999]
    if wrong == "pole":
        f = dp.lti_filter(transfer=(b, [1.0, -0.999]))(lambda x: ss.lfilter(b, a, x))
    else:
        f = dp.lti_filter(transfer=(b, a))(lambda x: 1.01 * ss.lfilter(b, a, x))
    report = dp.verify(f, max_examples=50, raise_on_failure=False)
    assert "transfer function" in [fl.relation for fl in report.failures]


@pytest.mark.parametrize("b, a", [
    ([1.0, -1.0], [1.0, -0.9999]),
    ([1.0], np.poly([0.999, 0.999])),  # double pole: h[n] grows like n r^n first
    ([1.0], np.real(np.poly(0.99 * np.exp(1j * np.array([0.3, -0.3, 0.3, -0.3]))))),
    tuple(ss.butter(4, 0.2)),
])
def test_tail_l1_bound_is_an_upper_bound(b, a):
    from dsp_props.core.iir import tail_l1_bound

    n = 2000
    impulse = np.zeros(400_000)
    impulse[0] = 1.0
    true_tail = np.sum(np.abs(ss.lfilter(b, a, impulse)[n:]))
    bound = tail_l1_bound(b, a, n)
    assert true_tail <= bound <= 10 * true_tail + 1e-300


def test_tail_l1_bound_is_zero_for_fir():
    from dsp_props.core.iir import tail_l1_bound

    assert tail_l1_bound([0.25, 0.5, 0.25], [1.0], 3) == 0.0


# --------------------------------------------------------------- structure="df2t" (lfilter)


def _random_stable(rng, order):
    """A stable (b, a) of the given order, poles of radius 0.5 to 0.999, a[0] in {0.25, 1, 4}."""
    half = order // 2
    r, th = rng.uniform(0.5, 0.999, half), rng.uniform(0, np.pi, half)
    poles = np.r_[r * np.exp(1j * th), r * np.exp(-1j * th), rng.uniform(-0.999, 0.999, order - 2 * half)]
    a0 = rng.choice([0.25, 1.0, 4.0])
    return a0 * rng.uniform(-1, 1, rng.integers(1, order + 3)), a0 * np.real(np.poly(poles))


def test_df2t_bound_holds_for_lfilter():
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(80):
        b, a = _random_stable(rng, int(rng.integers(2, 13)))
        tol = IIRTolerance(b, a, structure="df2t")
        w = float(np.max(np.angle(np.roots(a))))
        n = 256
        for x in (rng.standard_normal(n), np.cos(w * np.arange(n)), np.sign(rng.standard_normal(n))):
            err = norm(ss.lfilter(b, a, x) - longdouble_lfilter(b, a, x).astype(float))
            worst = max(worst, err / tol.bound(n, np.float64, norm(x)))
    assert worst < 0.1  # never reached: the bound is a worst case, the margin c = 4 comes on top


def test_df2t_is_tighter_on_usual_designs():
    for b, a in (ss.butter(4, 0.2), ss.cheby1(6, 1, 0.3), ss.ellip(4, 0.5, 40, 0.25)):
        df1, df2t = IIRTolerance(b, a).factor, IIRTolerance(b, a, structure="df2t").factor
        assert 1.5 < df1 / df2t < 4


def test_bound_does_not_depend_on_the_scaling_of_the_coefficients():
    # up to 0.1.4 the direct form I bound used the gain of 1/A(z) before normalisation by a[0]
    b, a = ss.butter(3, 0.3)
    one = IIRTolerance(b, a).factor
    assert np.isclose(IIRTolerance(4 * b, 4 * a).factor, one, rtol=1e-9)
    assert np.isclose(IIRTolerance(b / 4, a / 4).factor, one, rtol=1e-9)
    # df2t also counts the rounding of the division by a[0], absent when a[0] = 1
    assert IIRTolerance(4 * b, 4 * a, structure="df2t").factor > IIRTolerance(b, a, structure="df2t").factor


@pytest.mark.full_examples
def test_df2t_tolerance_passes_lfilter_and_catches_single_precision():
    b, a = ss.cheby1(6, 1, 0.3)
    tol = IIRTolerance(b, a, structure="df2t")
    good = dp.lti_filter(transfer=(b, a), tolerance=tol)(lambda x: ss.lfilter(b, a, x))
    assert dp.verify(good, max_examples=100, seed=0).ok

    def single(x):
        return ss.lfilter(b.astype(np.float32), a.astype(np.float32), np.asarray(x, np.float32)).astype(float)

    bad = dp.lti_filter(transfer=(b, a), tolerance=tol)(single)
    assert not dp.verify(bad, max_examples=100, seed=0, raise_on_failure=False).ok


def test_unknown_structure_is_refused():
    with pytest.raises(ValueError, match="structure"):
        IIRTolerance([1.0], [1.0, -0.5], structure="sos")
