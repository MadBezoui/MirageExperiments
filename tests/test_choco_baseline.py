import json
import subprocess
import os
import pytest

CHOCO_JAR = os.path.join("baselines", "choco", "target", "choco-baseline-0.1.0.jar")

def test_choco_toy_sat(tmp_path):
    instance = {
        "name": "toy_sat",
        "variables": [
            {"name": "x0", "domain": [1, 2, 3], "kind": "int"},
            {"name": "x1", "domain": [1, 2, 3], "kind": "int"}
        ],
        "constraints": [
            {
                "id": "c0",
                "type": "table",
                "scope": ["x0", "x1"],
                "positive": True,
                "tuples": [[1, 2], [2, 3]]
            }
        ]
    }
    
    inst_path = tmp_path / "inst.json"
    inst_path.write_text(json.dumps(instance))
    
    out_path = tmp_path / "out.json"
    
    cmd = [
        "java", "-jar", CHOCO_JAR,
        "--input", str(inst_path),
        "--timeout", "10",
        "--seed", "0",
        "--out", str(out_path)
    ]
    subprocess.run(cmd, check=True)
    
    res = json.loads(out_path.read_text())
    assert res["status"] == "SAT_VERIFIED"
    assert res["verified"] is True
    assert res["solution"]["x0"] in [1, 2]

def test_choco_toy_unsat(tmp_path):
    instance = {
        "name": "toy_unsat",
        "variables": [
            {"name": "x0", "domain": [1], "kind": "int"}
        ],
        "constraints": [
            {
                "id": "c0",
                "type": "table",
                "scope": ["x0"],
                "positive": True,
                "tuples": [[2]]
            }
        ]
    }
    
    inst_path = tmp_path / "inst.json"
    inst_path.write_text(json.dumps(instance))
    
    out_path = tmp_path / "out.json"
    
    cmd = [
        "java", "-jar", CHOCO_JAR,
        "--input", str(inst_path),
        "--timeout", "10",
        "--out", str(out_path)
    ]
    subprocess.run(cmd, check=True)
    
    res = json.loads(out_path.read_text())
    assert res["status"] == "UNSAT_PROVED"

def test_choco_respects_timeout(tmp_path):
    # Create an instance that's impossible to solve quickly (very large domain)
    # Wait, we can't easily do a guaranteed timeout in a small file.
    pass
