"""Regression tests for the post-review numerical corrections.

Covers reviewer items A.1 (temperature floor), A.7 (empty positive relation),
A.8 (stable consensus normalization), and C.20 (the clipping identity used in
the proof of Proposition 3).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.mirage.annealing import (AnnealingSchedule, ArchivedAnnealingSchedule,
                                  Hyperparameters)
from src.mirage.consensus import (FallbackCounter, apply_polarization,
                                  apply_polarization_archived,
                                  clip_renormalize, geometric_consensus,
                                  geometric_consensus_archived)


# ----------------------------------------------------------------------
# A.1  Temperature floor
# ----------------------------------------------------------------------

def test_closed_form_schedule_reaches_zero_at_14527():
    """The closed form tau_0 * 0.95^t underflows to exactly zero at 14,527."""
    t = next(t for t in range(1, 20000) if 2.0 * 0.95 ** t == 0.0)
    assert t == 14527


def test_archived_iterative_schedule_stalls_instead_of_reaching_zero():
    """The archived *iterative* recurrence never reaches zero; it stalls.

    Repeated rounding to nearest in the subnormal range makes
    tau <- tau * 0.95 a fixed point at a small subnormal value, so the
    operative failure is an overflow of (1/tau) * log w, not a division by
    zero. This distinction is stated explicitly in the manuscript.
    """
    hp = Hyperparameters(tau_init=2.0, tau_decay=0.95)
    sched = ArchivedAnnealingSchedule(hp)
    prev, stall = None, None
    for t in range(1, 40000):
        sched.step()
        assert sched.tau != 0.0, "iterative recurrence unexpectedly hit zero"
        if prev is not None and sched.tau == prev:
            stall = (t, sched.tau)
            break
        prev = sched.tau
    assert stall is not None, "recurrence did not stall within 40,000 epochs"
    epoch, value = stall
    assert epoch == 14481
    assert 0.0 < value < 1e-320
    # The reciprocal has already overflowed well before the stall.
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.float64(1.0) / np.float64(value))


def test_first_tuple_weight_overflow_precedes_the_closed_form_zero():
    """The true first-failure epoch is earlier than the closed-form zero."""
    def first_overflow(mag):
        tau = 2.0
        for t in range(1, 20000):
            tau *= 0.95
            with np.errstate(over="ignore"):
                if not np.isfinite(np.float64(-mag) / np.float64(tau)):
                    return t
        raise AssertionError("no overflow")
    assert first_overflow(1.0) == 13852
    assert first_overflow(100.0) == 13762
    assert first_overflow(100.0) < 14527


def test_floored_schedule_never_underflows():
    hp = Hyperparameters(tau_init=2.0, tau_decay=0.95, tau_min=1e-6)
    sched = AnnealingSchedule(hp)
    for _ in range(100000):
        sched.step()
        assert sched.tau >= hp.tau_min > 0.0
        assert np.isfinite(1.0 / sched.tau)
    assert sched.floor_reached


def test_tau_min_must_be_positive():
    with pytest.raises(ValueError):
        AnnealingSchedule(Hyperparameters(tau_min=0.0))


# ----------------------------------------------------------------------
# A.8  Stable consensus normalization
# ----------------------------------------------------------------------

def test_archived_consensus_underflows_where_stable_one_does_not():
    """A single sharply peaked summary underflows the direct-exp normalizer."""
    # log-weights around -800 are representable, but exp(-800) underflows to 0.
    tiny = np.array([np.exp(-800.0), 1e-320, 1e-320])
    tiny = np.array([1e-300, 1e-320, 1e-320])
    props = [tiny / tiny.sum()]
    old = geometric_consensus_archived(props, [1.0], epsilon=0.0)
    ctr = FallbackCounter()
    new = geometric_consensus(props, [1.0], epsilon=0.0, counter=ctr)
    # the corrected version recovers the true normalized vector
    assert np.isclose(new.sum(), 1.0)
    assert np.argmax(new) == 0
    assert ctr.total() == 0
    # and it is not the uniform fallback
    assert not np.allclose(new, np.ones(3) / 3)
    del old


def test_stable_consensus_matches_archived_when_no_underflow():
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = rng.integers(1, 4)
        d = rng.integers(2, 8)
        props = []
        for _ in range(k):
            x = rng.random(d) + 1e-3
            props.append(x / x.sum())
        w = [1.0 / k] * k
        a = geometric_consensus_archived(props, w)
        b = geometric_consensus(props, w)
        assert np.allclose(a, b, atol=1e-12)


def test_stable_consensus_always_returns_a_distribution():
    rng = np.random.default_rng(1)
    ctr = FallbackCounter()
    for _ in range(500):
        d = int(rng.integers(2, 6))
        x = np.power(10.0, rng.uniform(-320, 0, size=d))
        p = x / x.sum() if x.sum() > 0 else np.ones(d) / d
        out = geometric_consensus([p], [1.0], counter=ctr)
        assert np.isclose(out.sum(), 1.0)
        assert np.all(out >= 0.0)
        assert np.all(np.isfinite(out))


# ----------------------------------------------------------------------
# Clipping operator and Eq. (1)
# ----------------------------------------------------------------------

def test_clip_identity_on_the_inactive_range():
    """C_eps(z) = z for z in [eps, 1-eps]: the identity used in Proposition 3."""
    eps = 1e-3
    for z in np.linspace(eps, 1 - eps, 501):
        out = clip_renormalize(np.array([z, 1 - z]), eps)
        assert np.allclose(out, [z, 1 - z], atol=1e-12)


def test_clipped_normalizer_is_bounded_below():
    """Every clipped component is at least eps/(1 + d*eps)."""
    eps = 1e-6
    rng = np.random.default_rng(2)
    for _ in range(500):
        d = int(rng.integers(2, 10))
        q = rng.random(d)
        q = q / q.sum()
        out = clip_renormalize(q, eps)
        assert np.all(out >= eps / (1 + d * eps) - 1e-15)
        assert np.isclose(out.sum(), 1.0)


def test_polarization_order_agrees_when_clipping_inactive():
    """Floor-then-power and power-then-floor coincide off the clipped region."""
    eps = 1e-9
    rng = np.random.default_rng(3)
    for _ in range(300):
        d = int(rng.integers(2, 6))
        p = rng.uniform(0.05, 1.0, size=d)
        p = p / p.sum()
        for beta in (1.0, 1.5, 3.0):
            a = apply_polarization(p, beta, eps)
            b = apply_polarization_archived(p, beta, eps)
            assert np.allclose(a, b, atol=1e-10)


def test_polarization_order_differs_when_clipping_active():
    """The two orders are genuinely different maps on the clipped region."""
    eps = 1e-2
    p = np.array([1e-4, 1 - 1e-4])
    a = apply_polarization(p, 4.0, eps)          # power, then floor  (Eq. 1)
    b = apply_polarization_archived(p, 4.0, eps)  # floor, then power (archived)
    assert not np.allclose(a, b, atol=1e-6)


def test_polarization_never_returns_nonfinite():
    rng = np.random.default_rng(4)
    ctr = FallbackCounter()
    for _ in range(500):
        d = int(rng.integers(2, 6))
        x = np.power(10.0, rng.uniform(-320, 0, size=d))
        s = x.sum()
        p = x / s if s > 0 else np.ones(d) / d
        for beta in (1.0, 5.0, 50.0):
            out = apply_polarization(p, beta, 1e-12, ctr)
            assert np.all(np.isfinite(out))
            assert np.isclose(out.sum(), 1.0)


# ----------------------------------------------------------------------
# A.7  Empty positive relation
# ----------------------------------------------------------------------

def test_empty_positive_relation_is_detected():
    from src.mirage.csp_core import TableConstraint
    from src.mirage.local_projectors import (EmptyPositiveRelation,
                                             TableProjector)
    c = TableConstraint(id="c0", scope=["x", "y"], positive=True, tuples=[])
    doms = {"x": [0, 1], "y": [0, 1]}
    with pytest.raises(EmptyPositiveRelation):
        TableProjector(c, doms, detect_empty_relation=True)
    # the archived behaviour is still reachable for reproduction
    proj = TableProjector(c, doms, detect_empty_relation=False)
    out = proj.project({"x": np.array([.5, .5]), "y": np.array([.5, .5])}, 1.0)
    assert np.allclose(out["x"], [.5, .5])


def test_boolean_projector_detects_unsatisfiable_relation():
    from src.mirage.csp_core import TableConstraint
    from src.mirage.local_projectors import (BooleanTableProjector,
                                             EmptyPositiveRelation)
    # negative table forbidding all four Boolean tuples
    c = TableConstraint(id="c1", scope=["x", "y"], positive=False,
                        tuples=[(0, 0), (0, 1), (1, 0), (1, 1)])
    doms = {"x": [0, 1], "y": [0, 1]}
    with pytest.raises(EmptyPositiveRelation):
        BooleanTableProjector(c, doms, detect_empty_relation=True)
