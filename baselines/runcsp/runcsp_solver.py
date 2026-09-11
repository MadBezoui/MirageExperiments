"""RUN-CSP baseline: faithful PyTorch reimplementation of the recurrent
unsupervised network of Toenshoff, Ritzert, Wolf & Grohe,
"Graph Neural Networks for Maximum Constraint Satisfaction" (Frontiers in AI, 2021).

Scope / honesty notes (these MUST be mirrored in the paper's protocol appendix):

  * RUN-CSP is defined for *binary* CSPs (Max-2-CSP): every constraint relates
    at most two variables through an explicit relation matrix. Instances in our
    suite containing any non-table constraint or any table of arity > 2 are
    OUTSIDE the method's fragment and are reported with status
    ``UNSUPPORTED_FRAGMENT`` (they count as not-solved for RUN-CSP but the paper
    reports the supported-fragment size explicitly).
  * The original work trains one network per constraint-language on a
    distribution of instances. Our suite has heterogeneous domains and
    languages, so we use the standard transductive adaptation: the network is
    trained *unsupervised on the given instance itself* (the loss never sees a
    solution), with random restarts until the deadline.
  * RUN-CSP is incomplete: it can never prove UNSAT. Statuses are
    SAT_VERIFIED (assignment re-checked by the independent verifier),
    UNKNOWN_TIMEOUT, UNSUPPORTED_FRAGMENT, or ERROR.

Usage:
    python baselines/runcsp/runcsp_solver.py --input data/normalized/Foo.json \
        --seed 0 --timeout 60
"""  # noqa: E501
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Fragment check + instance compilation
# --------------------------------------------------------------------------- #

def _compile_binary_csp(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Compile a normalized instance into RUN-CSP form, or None if outside
    the binary-table fragment."""
    var_names = [v["name"] for v in data["variables"]]
    vidx = {n: i for i, n in enumerate(var_names)}
    domains = [[int(x) for x in v["domain"]] for v in data["variables"]]
    d_max = max(len(d) for d in domains)
    val2i = [{val: i for i, val in enumerate(dom)} for dom in domains]

    edges: List[Tuple[int, int, int]] = []
    unary: List[Tuple[int, List[int]]] = []
    rel_key_to_idx: Dict[bytes, int] = {}
    relations: List[List[List[int]]] = []

    for c in data["constraints"]:
        if c["type"] != "table":
            return None
        scope = c["scope"]
        if len(scope) > 2:
            return None
        positive = bool(c.get("positive", True))
        tuples = [tuple(int(x) for x in t) for t in c["tuples"]]

        if len(scope) == 1:
            u = vidx[scope[0]]
            mask = [0] * d_max
            if positive:
                for (val,) in tuples:
                    if val in val2i[u]:
                        mask[val2i[u][val]] = 1
            else:
                for i in range(len(domains[u])):
                    mask[i] = 1
                for (val,) in tuples:
                    if val in val2i[u]:
                        mask[val2i[u][val]] = 0
            unary.append((u, mask))
            continue

        u, v = vidx[scope[0]], vidx[scope[1]]
        # build d_max x d_max allowed matrix (padding rows/cols stay 0)
        M = [[0] * d_max for _ in range(d_max)]
        if positive:
            for (a, b) in tuples:
                if a in val2i[u] and b in val2i[v]:
                    M[val2i[u][a]][val2i[v][b]] = 1
        else:
            for i in range(len(domains[u])):
                for j in range(len(domains[v])):
                    M[i][j] = 1
            for (a, b) in tuples:
                if a in val2i[u] and b in val2i[v]:
                    M[val2i[u][a]][val2i[v][b]] = 0
        key = bytes(bytearray(x for row in M for x in row))
        if key not in rel_key_to_idx:
            rel_key_to_idx[key] = len(relations)
            relations.append(M)
        edges.append((u, v, rel_key_to_idx[key]))

    if not edges:
        return None  # degenerate; let complete solvers handle it
    return {"var_names": var_names, "domains": domains, "d_max": d_max,
            "edges": edges, "unary": unary, "relations": relations}


# --------------------------------------------------------------------------- #
# The network (faithful to the RUN-CSP architecture)
# --------------------------------------------------------------------------- #

def _run_runcsp(compiled: Dict[str, Any], data: Dict[str, Any], seed: int,
                deadline: float, state_dim: int = 128, t_max: int = 40,
                lr: float = 2e-3) -> Dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn as nn

    # independent verifier (soundness parity with MIRAGE-R)
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from experiments.validation.loader import reconstruct
    from src.mirage.verifier import verify_assignment

    instance = reconstruct(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = len(compiled["var_names"])
    d = compiled["d_max"]
    domains = compiled["domains"]

    # domain mask: True where the padded value index is a real domain value
    dom_mask = torch.zeros(n, d, dtype=torch.bool, device=device)
    for i, dom in enumerate(domains):
        dom_mask[i, :len(dom)] = True
    # unary constraints tighten the mask
    for (u, mask) in compiled["unary"]:
        m = torch.tensor(mask, dtype=torch.bool, device=device)
        dom_mask[u] &= m
        if not bool(dom_mask[u].any()):
            return {"status": "UNSAT_TRIVIAL_DOMAIN_WIPEOUT"}

    rels = torch.stack([torch.tensor(M, dtype=torch.float32, device=device)
                        for M in compiled["relations"]])          # (R, d, d)
    eu = torch.tensor([e[0] for e in compiled["edges"]], device=device)
    ev = torch.tensor([e[1] for e in compiled["edges"]], device=device)
    er = torch.tensor([e[2] for e in compiled["edges"]], device=device)
    n_edges = len(compiled["edges"])
    R = rels.shape[0]

    class RunCSPNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = nn.LSTMCell(state_dim, state_dim)
            # one message MLP per relation *type* and direction, as in RUN-CSP
            self.msg_fwd = nn.ModuleList(
                [nn.Linear(state_dim, state_dim) for _ in range(R)])
            self.msg_bwd = nn.ModuleList(
                [nn.Linear(state_dim, state_dim) for _ in range(R)])
            self.out = nn.Linear(state_dim, d)
            self.init = nn.Parameter(torch.randn(1, state_dim) * 0.1)

        def soft_assignment(self, h):
            logits = self.out(h)
            logits = logits.masked_fill(~dom_mask, -1e9)
            return torch.softmax(logits, dim=-1)

        def forward(self):
            h = self.init.expand(n, -1).contiguous()
            h = h + 0.01 * torch.randn(n, state_dim, device=device)
            c = torch.zeros_like(h)
            losses, softs = [], []
            for _ in range(t_max):
                agg = torch.zeros(n, state_dim, device=device)
                cnt = torch.zeros(n, 1, device=device)
                for r in range(R):
                    sel = (er == r)
                    if not bool(sel.any()):
                        continue
                    u_r, v_r = eu[sel], ev[sel]
                    m_uv = self.msg_fwd[r](h[u_r])   # message u -> v
                    m_vu = self.msg_bwd[r](h[v_r])   # message v -> u
                    agg.index_add_(0, v_r, m_uv)
                    agg.index_add_(0, u_r, m_vu)
                    ones = torch.ones(int(sel.sum()), 1, device=device)
                    cnt.index_add_(0, v_r, ones)
                    cnt.index_add_(0, u_r, ones)
                agg = agg / cnt.clamp(min=1.0)
                h, c = self.cell(agg, (h, c))
                x = self.soft_assignment(h)
                softs.append(x)
                # phi(u,v) = x_u^T M_r x_v  = P[constraint satisfied]
                xu, xv = x[eu], x[ev]                       # (E, d)
                Mx = torch.bmm(rels[er], xv.unsqueeze(-1)).squeeze(-1)
                phi = (xu * Mx).sum(-1).clamp(1e-8, 1.0)    # (E,)
                losses.append(-torch.log(phi).mean())
            # discounted sum over time as in the original (kappa^(T-t))
            T = len(losses)
            loss = sum((0.95 ** (T - 1 - t)) * losses[t] for t in range(T))
            return loss, softs

    best: Dict[str, Any] = {"status": "UNKNOWN_TIMEOUT", "restarts": 0,
                            "train_steps": 0, "n_edges": n_edges}
    restart = 0
    while time.time() < deadline:
        restart += 1
        torch.manual_seed(seed * 1009 + restart)
        np.random.seed((seed * 1009 + restart) % (2**31))
        net = RunCSPNet().to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _step in range(200):
            if time.time() >= deadline:
                break
            opt.zero_grad()
            loss, softs = net()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            best["train_steps"] += 1
            # decode the final and mid unrolled soft assignments; verify
            with torch.no_grad():
                for x in (softs[-1], softs[len(softs) // 2]):
                    hard = x.argmax(dim=-1).cpu().numpy()
                    assignment = {}
                    ok = True
                    for i, name in enumerate(compiled["var_names"]):
                        k = int(hard[i])
                        if k >= len(domains[i]):
                            ok = False
                            break
                        assignment[name] = domains[i][k]
                    if not ok:
                        continue
                    if verify_assignment(instance, assignment):
                        best.update(status="SAT_VERIFIED", restarts=restart)
                        best["solution"] = assignment
                        return best
        best["restarts"] = restart
    return best


# --------------------------------------------------------------------------- #

def solve_instance(json_path: str, seed: int = 0, timeout: float = 300.0,
                   **_ignored) -> Dict[str, Any]:
    name = os.path.basename(json_path)
    t0 = time.time()
    base = {"instance": name, "solver": "runcsp", "seed": seed,
            "violations": -1, "branches": 0, "conflicts": 0, "error": ""}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        compiled = _compile_binary_csp(data)
        if compiled is None:
            return {**base, "status": "UNSUPPORTED_FRAGMENT",
                    "time": time.time() - t0,
                    "error": "outside RUN-CSP binary-table fragment"}
        res = _run_runcsp(compiled, data, seed, deadline=t0 + timeout)
        status = res.pop("status")
        if status == "UNSAT_TRIVIAL_DOMAIN_WIPEOUT":
            # a unary table wiped out a domain: the instance is UNSAT, but
            # RUN-CSP cannot certify UNSAT -- report honestly as unsupported
            return {**base, "status": "UNSUPPORTED_FRAGMENT",
                    "time": time.time() - t0, "error": "unary domain wipeout"}
        out = {**base, "status": status, "time": time.time() - t0}
        out.update({k: v for k, v in res.items() if k != "solution"})
        if "solution" in res:
            out["solution"] = res["solution"]
            out["violations"] = 0
        return out
    except Exception as e:  # noqa: BLE001
        return {**base, "status": "ERROR", "time": time.time() - t0,
                "error": str(e)[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()
    rec = solve_instance(args.input, args.seed, args.timeout)
    rec.pop("solution", None)
    print(json.dumps(rec))
    return 0 if rec["status"] != "ERROR" else 1


if __name__ == "__main__":
    sys.exit(main())
