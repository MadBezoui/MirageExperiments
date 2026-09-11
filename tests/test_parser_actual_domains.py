import pytest
from src.mirage.csp_core import Variable, TableConstraint, SumConstraint, CountConstraint, ElementConstraint, AllDifferentConstraint

def test_actual_domains():
    v1 = Variable("v1", [1, 3, 5])
    assert v1.val2idx[3] == 1
    assert v1.idx2val[1] == 3

def test_table_constraint():
    c = TableConstraint("t1", ["v1", "v2"], positive=True, tuples=[(1, 2), (3, 4)])
    assert c.is_satisfied({"v1": 1, "v2": 2})
    assert not c.is_satisfied({"v1": 1, "v2": 4})

def test_sum_constraint():
    c = SumConstraint("s1", ["v1", "v2"], [1, -1], "eq", 0)
    assert c.is_satisfied({"v1": 5, "v2": 5})
    assert not c.is_satisfied({"v1": 5, "v2": 3})

def test_count_constraint():
    c = CountConstraint("c1", ["v1", "v2", "v3"], [1], "eq", 2)
    assert c.is_satisfied({"v1": 1, "v2": 1, "v3": 5})
    assert not c.is_satisfied({"v1": 1, "v2": 5, "v3": 5})

def test_element_constraint():
    c = ElementConstraint("e1", ["v1", "v2", "v3", "idx", "val"], "idx", "val", array_vars=["v1", "v2", "v3"])
    # idx=1 means we pick v2. v2=10. val=10 -> True
    assert c.is_satisfied({"idx": 1, "val": 10, "v1": 5, "v2": 10, "v3": 15})
    assert not c.is_satisfied({"idx": 1, "val": 15, "v1": 5, "v2": 10, "v3": 15})
    assert not c.is_satisfied({"idx": 5, "val": 10, "v1": 5, "v2": 10, "v3": 15}) # out of bounds

def test_element_constraint_values():
    c = ElementConstraint("e2", [], "idx", "val", array_values=[5, 10, 15])
    assert c.is_satisfied({"idx": 1, "val": 10})
    assert not c.is_satisfied({"idx": 1, "val": 5})

def test_alldiff_constraint():
    c = AllDifferentConstraint("a1", ["v1", "v2", "v3"])
    assert c.is_satisfied({"v1": 1, "v2": 2, "v3": 3})
    assert not c.is_satisfied({"v1": 1, "v2": 2, "v3": 1})
