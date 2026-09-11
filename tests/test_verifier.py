import pytest
from src.mirage.csp_core import CSPInstance, Variable, TableConstraint
from src.mirage.verifier import verify_assignment, count_violations, violated_constraints

def test_verifier():
    v1 = Variable("v1", [1, 2])
    v2 = Variable("v2", [3, 4])
    c1 = TableConstraint("c1", ["v1", "v2"], True, [(1, 3), (2, 4)])
    
    inst = CSPInstance("test", "test.xml", {"v1": v1, "v2": v2}, [c1], {})
    
    assert verify_assignment(inst, {"v1": 1, "v2": 3}) is True
    assert verify_assignment(inst, {"v1": 1, "v2": 4}) is False
    assert count_violations(inst, {"v1": 1, "v2": 4}) == 1
    assert violated_constraints(inst, {"v1": 1, "v2": 4}) == ["c1"]
