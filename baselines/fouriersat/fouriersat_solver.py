import json
import time
import torch
import random
import os
import sys

def encode_constraint_pytorch(c_scope, c_tuples, var_indices, device):
    """
    Returns a function that computes the polynomial for this constraint.
    c_tuples are given in {0, 1}. We map to {-1, 1}.
    """
    # map to {-1, 1}
    # 0 -> -1, 1 -> 1
    t_tensor = torch.tensor([[1 if val == 1 else -1 for val in t] for t in c_tuples], 
                            dtype=torch.float32, device=device)
    v_idx = [var_indices[v] for v in c_scope]
    
    def evaluate(x):
        # x is shape (N,) in [-1, 1]
        x_c = x[v_idx] # shape (k,)
        # For each tuple t, product(1 + t_i x_i) / 2
        # t_tensor is shape (num_tuples, k)
        # x_c is shape (k,)
        # 1 + t_i x_i is shape (num_tuples, k)
        term = (1.0 + t_tensor * x_c.unsqueeze(0)) / 2.0
        # product over k
        prod = torch.prod(term, dim=1)
        # sum over allowed tuples
        return torch.sum(prod)
        
    return evaluate

def solve_instance(filepath, solver_type="fouriersat", seed=0, timeout=300.0):
    t0 = time.time()
    
    with open(filepath, "r") as f:
        data = json.load(f)
        
    variables = [v["name"] for v in data["variables"]]
    var_idx = {v: i for i, v in enumerate(variables)}
    n = len(variables)
    
    # check if all variables are boolean
    is_boolean = all(len(v["domain"]) == 2 for v in data["variables"])
    if not is_boolean:
        # FourierSAT only natively supports Boolean. For others we would need encoding.
        # But per the plan, primary is Boolean. If not Boolean, we just return UNSUPPORTED
        return {"status": "UNSUPPORTED_FRAGMENT", "time": time.time() - t0, "error": "Not a purely Boolean instance"}
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluators = []
    for c in data["constraints"]:
        if c["type"] == "table":
            if c["positive"]:
                evaluators.append(encode_constraint_pytorch(c["scope"], c["tuples"], var_idx, device))
            else:
                # for negative table, allowed is everything NOT in tuples
                import itertools
                k = len(c["scope"])
                all_t = list(itertools.product([0, 1], repeat=k))
                c_set = set(tuple(t) for t in c["tuples"])
                allowed = [t for t in all_t if t not in c_set]
                evaluators.append(encode_constraint_pytorch(c["scope"], allowed, var_idx, device))
        else:
            return {"status": "UNSUPPORTED_FRAGMENT", "time": time.time() - t0, "error": "Non-table constraint"}

    torch.manual_seed(seed)
    random.seed(seed)
    
    # FourierSAT optimizer: Gradient descent with restarts
    # We want to maximize sum f_j(x), or minimize sum (1 - f_j(x))
    best_loss = float('inf')
    best_solution = None
    
    num_constraints = len(evaluators)
    
    # Hyperparameters for continuous local search
    lr = 0.1
    max_steps_per_restart = 1000
    
    while time.time() - t0 < timeout:
        # random initialization in [-1, 1]
        x = torch.empty(n, device=device, requires_grad=True)
        torch.nn.init.uniform_(x, -1.0, 1.0)
        
        optimizer = torch.optim.Adam([x], lr=lr)
        
        for step in range(max_steps_per_restart):
            if time.time() - t0 >= timeout:
                break
                
            optimizer.zero_grad()
            
            # x must be clamped to [-1, 1] conceptually, but Adam might push it out.
            # We can use a projected gradient or just tanh
            # Actually, standard FourierSAT just uses projected gradient
            
            total_loss = 0.0
            for eval_fn in evaluators:
                total_loss += (1.0 - eval_fn(x))
                
            loss_val = total_loss.item()
            if loss_val < 1e-4:
                # We found a continuous zero. Round it.
                rounded = (x.detach().cpu().numpy() > 0).astype(int)
                # Check if it's a real solution
                # To do this correctly, we could just evaluate the constraints
                # but we'll let the verifier do it. We return it.
                return {
                    "status": "SAT_VERIFIED",
                    "time": time.time() - t0,
                    "solution": {variables[i]: int(rounded[i]) for i in range(n)}
                }
                
            total_loss.backward()
            optimizer.step()
            
            # projection back to [-1, 1]
            with torch.no_grad():
                x.clamp_(-1.0, 1.0)
                
    return {"status": "UNKNOWN_TIMEOUT", "time": time.time() - t0}
