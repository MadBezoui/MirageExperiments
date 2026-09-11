"""GPU-batched throughput experiment (revision P2.4).

Cashes out the "differentiable / accelerator-friendly" motivation with a
measured number instead of a promise: the MIRAGE-R epoch for table constraints
is expressed as dense tensor ops (gather -> log-sum -> softmax over tuples ->
scatter-add), batched over ALL projectors at once, and timed per epoch on

  * numpy      : the per-projector loop as shipped in src/mirage (reference),
  * torch-cpu  : the vectorized batched epoch on CPU,
  * torch-cuda : the same on GPU (fp32), if CUDA is available.

The instance grid is synthetic random binary table CSPs at increasing scale so
the scaling exponent is clean (real instances confound size with structure).
Semantics of one epoch are IDENTICAL across backends (same tempered-projection
math); we verify marginals agree to 1e-4 on the smallest size before timing.

Outputs results/raw/revision_gpu/gpu_throughput.jsonl and (via aggregate.py)
the wall-clock/epoch scaling figure + macros. If no CUDA device exists, the
experiment still runs numpy vs torch-cpu and the paper text (macro-driven)
reports vectorized-CPU speedup only.

Usage:
    PYTHONPATH=. python -m experiments.gpu_throughput \
        --out-dir results/raw/revision_gpu --epochs 20
"""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np

SIZES = [  # (n_vars, n_constraints, domain, tuples_per_constraint)
    (200, 400, 8, 16),
    (500, 1000, 8, 16),
    (1000, 2000, 8, 16),
    (2000, 4000, 8, 16),
    (5000, 10000, 8, 16),
    (10000, 20000, 8, 16),
    (20000, 40000, 8, 16),
]


def make_instance(n, m, d, k, rng):
    """Random binary positive-table CSP; returns (scopes, tuples)."""
    scopes = rng.integers(0, n, size=(m, 2))
    same = scopes[:, 0] == scopes[:, 1]
    scopes[same, 1] = (scopes[same, 1] + 1) % n
    tuples = rng.integers(0, d, size=(m, k, 2))
    return scopes, tuples


# --------------------------------------------------------------------------- #
def epoch_numpy(p, scopes, tuples, tau):
    """Reference per-projector loop (mirrors src/mirage/local_projectors.py)."""
    n, d = p.shape
    m, k, _ = tuples.shape
    prop_sum = np.zeros_like(p)
    prop_cnt = np.zeros(n)
    for c in range(m):
        u, v = scopes[c]
        logw = (np.log(np.maximum(p[u][tuples[c, :, 0]], 1e-12))
                + np.log(np.maximum(p[v][tuples[c, :, 1]], 1e-12))) / tau
        w = np.exp(logw - logw.max())
        w /= w.sum()
        pu = np.zeros(d)
        pv = np.zeros(d)
        np.add.at(pu, tuples[c, :, 0], w)
        np.add.at(pv, tuples[c, :, 1], w)
        prop_sum[u] += np.log(np.maximum(pu, 1e-12))
        prop_sum[v] += np.log(np.maximum(pv, 1e-12))
        prop_cnt[u] += 1
        prop_cnt[v] += 1
    # geometric consensus
    out = np.exp(prop_sum / np.maximum(prop_cnt, 1)[:, None])
    out = np.where(prop_cnt[:, None] > 0, out, p)
    return out / out.sum(axis=1, keepdims=True)


def epoch_torch(p, scopes, tuples, tau):
    """Vectorized batched epoch: all m projectors in one shot."""
    import torch
    n, d = p.shape
    m, k, _ = tuples.shape
    pu = p[scopes[:, 0]]                                   # (m, d)
    pv = p[scopes[:, 1]]
    tu = tuples[:, :, 0]                                   # (m, k)
    tv = tuples[:, :, 1]
    logw = (torch.log(pu.gather(1, tu).clamp(min=1e-12))
            + torch.log(pv.gather(1, tv).clamp(min=1e-12))) / tau
    w = torch.softmax(logw, dim=1)                         # (m, k)
    mu = torch.zeros(m, d, device=p.device, dtype=p.dtype).scatter_add_(1, tu, w)
    mv = torch.zeros(m, d, device=p.device, dtype=p.dtype).scatter_add_(1, tv, w)
    prop_sum = torch.zeros(n, d, device=p.device, dtype=p.dtype)
    prop_sum.index_add_(0, scopes[:, 0], torch.log(mu.clamp(min=1e-12)))
    prop_sum.index_add_(0, scopes[:, 1], torch.log(mv.clamp(min=1e-12)))
    cnt = torch.zeros(n, device=p.device, dtype=p.dtype)
    ones = torch.ones(m, device=p.device, dtype=p.dtype)
    cnt.index_add_(0, scopes[:, 0], ones)
    cnt.index_add_(0, scopes[:, 1], ones)
    out = torch.exp(prop_sum / cnt.clamp(min=1).unsqueeze(1))
    out = torch.where((cnt > 0).unsqueeze(1), out, p)
    return out / out.sum(dim=1, keepdim=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--out-dir", default="results/raw/revision_gpu")
    args = ap.parse_args()

    import torch
    has_cuda = torch.cuda.is_available()
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "gpu_throughput.jsonl")
    rng = np.random.default_rng(0)

    # semantic agreement check on the smallest size ------------------------- #
    n, m, d, k = SIZES[0]
    scopes, tuples = make_instance(n, m, d, k, rng)
    p0 = rng.random((n, d))
    p0 = p0 / p0.sum(axis=1, keepdims=True)
    ref = epoch_numpy(p0.copy(), scopes, tuples, tau=2.0)
    tt = epoch_torch(torch.tensor(p0, dtype=torch.float64),
                     torch.tensor(scopes), torch.tensor(tuples),
                     tau=2.0).numpy()
    err = float(np.abs(ref - tt).max())
    assert err < 1e-4, f"backend semantics diverged: max err {err}"
    print(f"[gpu] semantic agreement OK (max err {err:.2e}); cuda={has_cuda}",
          flush=True)

    records = []
    with open(out_path, "w", buffering=1) as fh:
        for (n, m, d, k) in SIZES:
            scopes_np, tuples_np = make_instance(n, m, d, k, rng)
            p_np = rng.random((n, d))
            p_np = p_np / p_np.sum(axis=1, keepdims=True)

            backends = {}
            # numpy reference loop (skip at large scale: prohibitively slow)
            if n <= 5000:
                p = p_np.copy()
                t0 = time.perf_counter()
                for _ in range(max(args.epochs // 4, 2)):
                    p = epoch_numpy(p, scopes_np, tuples_np, tau=2.0)
                backends["numpy_loop"] = ((time.perf_counter() - t0)
                                          / max(args.epochs // 4, 2))

            for dev in (["cpu", "cuda"] if has_cuda else ["cpu"]):
                dt = torch.float32
                pt = torch.tensor(p_np, dtype=dt, device=dev)
                sc = torch.tensor(scopes_np, device=dev)
                tp = torch.tensor(tuples_np, device=dev)
                for _ in range(args.warmup_epochs):
                    pt = epoch_torch(pt, sc, tp, tau=2.0)
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.epochs):
                    pt = epoch_torch(pt, sc, tp, tau=2.0)
                if dev == "cuda":
                    torch.cuda.synchronize()
                backends[f"torch_{dev}"] = (time.perf_counter() - t0) / args.epochs

            rec = {"n_vars": n, "n_constraints": m, "domain": d,
                   "tuples_per_con": k, "cuda_available": has_cuda,
                   "sec_per_epoch": {b: round(v, 6) for b, v in backends.items()}}
            fh.write(json.dumps(rec) + "\n")
            records.append(rec)
            print(f"[gpu] n={n} m={m}: " + ", ".join(
                f"{b}={v*1e3:.2f}ms" for b, v in backends.items()), flush=True)

    print(f"[gpu] DONE -> {out_path}")


if __name__ == "__main__":
    main()
