import os
import json
import random
import itertools

def generate_random_boolean_table(scope, arity, density, sat_ratio=0.5):
    """Generate a random Boolean table constraint."""
    # sat_ratio is the ratio of satisfying assignments out of 2^arity
    num_allowed = int(sat_ratio * (1 << arity))
    if num_allowed == 0:
        num_allowed = 1
    if num_allowed == (1 << arity):
        num_allowed = (1 << arity) - 1
        
    all_tuples = list(itertools.product([0, 1], repeat=arity))
    allowed_tuples = random.sample(all_tuples, num_allowed)
    
    return {
        "id": f"c_{scope}",
        "type": "table",
        "scope": scope,
        "positive": True,
        "tuples": allowed_tuples
    }

def generate_instance(n, arity, density, seed):
    random.seed(seed)
    
    num_constraints = int(density * n)
    variables = [{"name": f"x{i}", "domain": [0, 1], "kind": "int"} for i in range(n)]
    
    constraints = []
    for c_idx in range(num_constraints):
        # randomly pick 'arity' variables
        scope_vars = random.sample(range(n), arity)
        scope = [f"x{i}" for i in scope_vars]
        # near satisfiability threshold for random k-SAT
        # A known phase transition for k-SAT is alpha_c = 2^k ln 2. 
        # For general boolean CSPs with p allowed tuples, it's roughly -ln(2) / ln(1 - p/2^k)
        # We will just pick a SAT ratio that makes it interesting.
        sat_ratio = 1.0 - (1.0 / (2.0 ** (arity / 2.0))) # arbitrary heuristic
        sat_ratio = max(0.1, min(0.9, sat_ratio))
        
        c = generate_random_boolean_table(scope, arity, density, sat_ratio)
        c["id"] = f"c{c_idx}"
        constraints.append(c)
        
    return {
        "name": f"bool_n{n}_k{arity}_d{density}_s{seed}",
        "source_path": "generator",
        "variables": variables,
        "constraints": constraints
    }

def main():
    out_dir = "data/normalized"
    os.makedirs(out_dir, exist_ok=True)
    
    arities = [2, 3, 4, 5, 8]
    densities = [0.5, 1.0, 2.0, 4.0]
    n_values = [50]
    num_instances = 20
    
    seed = 42
    for n in n_values:
        for k in arities:
            for d in densities:
                for i in range(num_instances):
                    inst = generate_instance(n, k, d, seed)
                    seed += 1
                    
                    filename = f"{inst['name']}.json"
                    with open(os.path.join(out_dir, filename), "w") as f:
                        json.dump(inst, f)
                        
    print("Generated Boolean CSPs.")

if __name__ == "__main__":
    main()
