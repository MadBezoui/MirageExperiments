"""Annealing schedules for MIRAGE-R.

Correction applied after the archived sweep (reviewer item A.1): the
temperature schedule now has a declared positive floor

    tau_t = max(tau_min, tau_0 * tau_decay^t),

so that 1/tau_t is always finite. The archived schedule was the unfloored
geometric decay tau_t = tau_0 * 0.95^t, which reaches the smallest positive
double at epoch 14,527 for tau_0 = 2 and then underflows to exactly zero,
after which the tuple log-weight (1/tau) * sum_i log p_i(t_i) is undefined and
the run no longer executes the intended algorithm.

The floor is a declared hyperparameter, not a silent guard: `tau_min` appears
in Hyperparameters and is reported in the protocol table.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Hyperparameters:
    epsilon: float = 1e-12
    gamma_init: float = 0.1
    gamma_decay: float = 0.99
    tau_init: float = 1.0
    tau_decay: float = 0.95
    tau_min: float = 1e-6             # declared temperature floor (item A.1)
    beta_init: float = 1.0
    beta_growth: float = 1.05
    beta_max: float = 5.0
    restart_policy: str = "luby"
    adaptive_region_budget: int = 100
    cycle_length_cap: int = 10
    tuple_cap: int = 10000
    decoding_frequency: int = 5
    max_epochs: int = 1000
    use_adaptive_regions: bool = False
    stagnation_window: int = 10
    timeout: float = 300.0
    strict_finite: bool = True        # assert finiteness of intermediates
    detect_empty_relation: bool = True  # item A.7
    seeds: list[int] = None


class AnnealingSchedule:
    """Geometric temperature decay with a floor, and capped polarization growth.

    The zero-temperature limit is *not* supported by the update: as tau -> 0+
    the tempered tuple measure converges to the uniform distribution over the
    argmax tuples of sum_i log p_i(t_i), which is a different (discrete-argmax)
    operator. Rather than silently switching operators, the schedule stops at
    tau_min and the run continues on the intended continuous map.
    """

    def __init__(self, hp: Hyperparameters):
        if hp.tau_min <= 0.0:
            raise ValueError("tau_min must be strictly positive")
        self.hp = hp
        self.t = 0
        self.gamma = hp.gamma_init
        self.tau = max(hp.tau_min, hp.tau_init)
        self.beta = hp.beta_init

    def step(self):
        self.t += 1
        self.gamma *= self.hp.gamma_decay
        self.tau = max(self.hp.tau_min, self.tau * self.hp.tau_decay)
        self.beta = min(self.hp.beta_max, self.beta * self.hp.beta_growth)
        if self.hp.strict_finite:
            assert self.tau >= self.hp.tau_min > 0.0, "temperature underflow"
            assert self.beta > 0.0, "non-positive polarization exponent"

    @property
    def floor_reached(self) -> bool:
        return self.tau <= self.hp.tau_min

    def as_dict(self) -> dict:
        return {"epoch": self.t, "tau": self.tau, "beta": self.beta,
                "gamma_ratio": self.beta / self.tau}


class ArchivedAnnealingSchedule(AnnealingSchedule):
    """The pre-correction schedule with no temperature floor.

    Retained only to reproduce the archived behaviour, including the underflow
    at epoch 14,527 for tau_0 = 2.
    """

    def __init__(self, hp: Hyperparameters):
        self.hp = hp
        self.t = 0
        self.gamma = hp.gamma_init
        self.tau = hp.tau_init
        self.beta = hp.beta_init

    def step(self):
        self.t += 1
        self.gamma *= self.hp.gamma_decay
        self.tau *= self.hp.tau_decay
        self.beta = min(self.hp.beta_max, self.beta * self.hp.beta_growth)
