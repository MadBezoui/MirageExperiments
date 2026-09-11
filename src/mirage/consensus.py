"""Geometric consensus and polarization, with numerically stable normalization.

Corrections applied after the archived sweep (reviewer items A.8, B.9):

1. The consensus normalizer is now computed with max subtraction
   (log-sum-exp), so that

       phat_i(a) = exp(l_i(a) - m_i) / sum_b exp(l_i(b) - m_i),
       m_i       = max_b l_i(b),

   which cannot underflow to an all-zero vector for finite inputs. The
   archived implementation evaluated exp(l_i(a)) directly and fell back to the
   uniform marginal whenever every component underflowed. That fallback is
   retained only as a genuine non-finite guard.

2. The clipping operator is applied *after* polarization, matching Eq. (1) of
   the manuscript. The archived implementation applied the floor to the
   consensus marginal *before* raising it to the power beta. The two agree
   whenever the floor is inactive (every component in [eps, 1-eps]), which
   includes the whole critical fixed-point continuum analyzed in the paper,
   but they are not identical maps in general.

3. Fallback events are counted rather than being silent, so an experiment can
   report how often the numerical guard fired.
"""
from __future__ import annotations

import numpy as np


class FallbackCounter:
    """Counts numerical guard activations over a run."""

    __slots__ = ("nonfinite_consensus", "nonfinite_polarization",
                 "underflow_consensus")

    def __init__(self):
        self.nonfinite_consensus = 0
        self.nonfinite_polarization = 0
        self.underflow_consensus = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "fallback_nonfinite_consensus": self.nonfinite_consensus,
            "fallback_nonfinite_polarization": self.nonfinite_polarization,
            "fallback_underflow_consensus": self.underflow_consensus,
        }

    def total(self) -> int:
        return (self.nonfinite_consensus + self.nonfinite_polarization
                + self.underflow_consensus)


def geometric_consensus(marginals_proposals: list[np.ndarray],
                        weights: list[float],
                        epsilon: float = 1e-12,
                        counter: "FallbackCounter | None" = None) -> np.ndarray:
    """Weighted geometric mean of factor summaries for one variable.

    Implements

        l_i(a)    = sum_c w_{c,i} log(max(pi_{c->i}(a), eps))
        phat_i(a) = softmax_a l_i(a)

    using max subtraction. For finite l_i the result is always a valid
    probability vector, so the uniform fallback fires only on genuinely
    non-finite input.
    """
    assert len(marginals_proposals) == len(weights)
    log_p = np.zeros(len(marginals_proposals[0]), dtype=float)
    for p, w in zip(marginals_proposals, weights):
        safe_p = np.clip(np.asarray(p, dtype=float), epsilon, 1.0)
        log_p += w * np.log(safe_p)

    if not np.all(np.isfinite(log_p)):
        if counter is not None:
            counter.nonfinite_consensus += 1
        return np.ones_like(log_p) / len(log_p)

    m = np.max(log_p)
    shifted = np.exp(log_p - m)
    total = np.sum(shifted)
    if not np.isfinite(total) or total <= 0.0:
        # Unreachable for finite log_p: the maximal component is exp(0) = 1,
        # so total >= 1. Retained as a defensive guard.
        if counter is not None:
            counter.underflow_consensus += 1
        return np.ones_like(log_p) / len(log_p)
    return shifted / total


def apply_polarization(p: np.ndarray, beta: float,
                       epsilon: float = 1e-12,
                       counter: "FallbackCounter | None" = None) -> np.ndarray:
    """Polarization followed by the clipping operator of Eq. (1).

        q_i(a)   = phat_i(a)^beta / sum_b phat_i(b)^beta
        p_i^+(a) = max(q_i(a), eps) / sum_b max(q_i(b), eps)

    Computed in the log domain so that large beta cannot underflow the
    polarization normalizer.
    """
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore"):
        log_p = np.log(np.clip(p, 0.0, 1.0))
    finite = np.isfinite(log_p)
    if not np.any(finite):
        if counter is not None:
            counter.nonfinite_polarization += 1
        return np.ones_like(p) / len(p)
    log_q = beta * log_p
    m = np.max(log_q[finite])
    q = np.where(finite, np.exp(log_q - m), 0.0)
    s = np.sum(q)
    if not np.isfinite(s) or s <= 0.0:
        if counter is not None:
            counter.nonfinite_polarization += 1
        return np.ones_like(p) / len(p)
    q = q / s
    # Clipping operator: floor applied AFTER polarization, then renormalize.
    clipped = np.maximum(q, epsilon)
    return clipped / np.sum(clipped)


def clip_renormalize(z: np.ndarray, epsilon: float) -> np.ndarray:
    """The clipping operator C_eps of the manuscript, exposed for testing."""
    c = np.maximum(np.asarray(z, dtype=float), epsilon)
    return c / np.sum(c)


# ----------------------------------------------------------------------
# Legacy behaviour, retained so that the archived runs remain reproducible.
# ----------------------------------------------------------------------

def geometric_consensus_archived(marginals_proposals, weights,
                                 epsilon: float = 1e-12) -> np.ndarray:
    """The pre-correction consensus: direct exp, uniform fallback on underflow.

    Kept only to reproduce the archived sweep; not used by the corrected
    solver.
    """
    log_p = np.zeros(len(marginals_proposals[0]), dtype=float)
    for p, w in zip(marginals_proposals, weights):
        log_p += w * np.log(np.clip(np.asarray(p, dtype=float), epsilon, 1.0))
    p_new = np.exp(log_p)
    s = np.sum(p_new)
    if s > 0:
        return p_new / s
    return np.ones_like(p_new) / len(p_new)


def apply_polarization_archived(p, beta: float,
                                epsilon: float = 1e-12) -> np.ndarray:
    """The pre-correction polarization: floor applied BEFORE the exponent."""
    safe_p = np.clip(np.asarray(p, dtype=float), epsilon, 1.0)
    p_pol = np.power(safe_p, beta)
    s = np.sum(p_pol)
    if s > 0:
        return p_pol / s
    return np.ones_like(p_pol) / len(p_pol)
