from dataclasses import dataclass, field
from typing import Any

@dataclass
class Variable:
    name: str
    domain: list[int]
    kind: str = "int"

    def __post_init__(self):
        self.val2idx = {val: idx for idx, val in enumerate(self.domain)}
        self.idx2val = {idx: val for idx, val in enumerate(self.domain)}

@dataclass
class Constraint:
    id: str
    scope: list[str]

    def is_satisfied(self, assignment_actual_values: dict[str, int]) -> bool:
        raise NotImplementedError

    def to_normalized_json(self) -> dict[str, Any]:
        raise NotImplementedError

@dataclass
class TableConstraint(Constraint):
    positive: bool
    tuples: list[tuple[int, ...]]

    def __post_init__(self):
        self.tuples_set = set(self.tuples)

    def is_satisfied(self, assignment_actual_values: dict[str, int]) -> bool:
        tup = tuple(assignment_actual_values[v] for v in self.scope)
        if self.positive:
            return tup in self.tuples_set
        else:
            return tup not in self.tuples_set

    def to_normalized_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "table",
            "scope": self.scope,
            "positive": self.positive,
            "tuples": [list(t) for t in self.tuples]
        }

@dataclass
class SumConstraint(Constraint):
    coeffs: list[int]
    operator: str
    rhs: int

    def is_satisfied(self, assignment_actual_values: dict[str, int]) -> bool:
        total = sum(c * assignment_actual_values[v] for c, v in zip(self.coeffs, self.scope))
        if self.operator == "=" or self.operator == "eq": return total == self.rhs
        if self.operator == "!=" or self.operator == "ne": return total != self.rhs
        if self.operator == "<" or self.operator == "lt": return total < self.rhs
        if self.operator == "<=" or self.operator == "le": return total <= self.rhs
        if self.operator == ">" or self.operator == "gt": return total > self.rhs
        if self.operator == ">=" or self.operator == "ge": return total >= self.rhs
        return False

    def to_normalized_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "sum",
            "scope": self.scope,
            "coeffs": self.coeffs,
            "operator": self.operator,
            "rhs": self.rhs
        }

@dataclass
class CountConstraint(Constraint):
    values: list[int]
    operator: str
    rhs: int

    def is_satisfied(self, assignment_actual_values: dict[str, int]) -> bool:
        count = sum(1 for v in self.scope if assignment_actual_values[v] in self.values)
        if self.operator == "=" or self.operator == "eq": return count == self.rhs
        if self.operator == "!=" or self.operator == "ne": return count != self.rhs
        if self.operator == "<" or self.operator == "lt": return count < self.rhs
        if self.operator == "<=" or self.operator == "le": return count <= self.rhs
        if self.operator == ">" or self.operator == "gt": return count > self.rhs
        if self.operator == ">=" or self.operator == "ge": return count >= self.rhs
        return False

    def to_normalized_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "count",
            "scope": self.scope,
            "values": self.values,
            "operator": self.operator,
            "rhs": self.rhs
        }

@dataclass
class ElementConstraint(Constraint):
    index_var: str
    value_var: str
    array_vars: list[str] = field(default_factory=list)
    array_values: list[int] = field(default_factory=list)

    def is_satisfied(self, assignment_actual_values: dict[str, int]) -> bool:
        idx = assignment_actual_values[self.index_var]
        # XCSP3 element arrays are usually 0-indexed or 1-indexed? We need to handle this carefully.
        # But XCSP3 standard says index variables can have arbitrary domain, but in <element>, the array is 0-indexed or explicitly indexed?
        # Typically the array is 0-indexed, meaning if idx = 0, it takes the first element.
        # Let's assume 0-indexed array by default.
        if idx < 0 or idx >= max(len(self.array_vars), len(self.array_values)):
            return False
        if self.array_vars:
            val = assignment_actual_values[self.array_vars[idx]]
        else:
            val = self.array_values[idx]
        return val == assignment_actual_values[self.value_var]

    def to_normalized_json(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "type": "element",
            "scope": self.scope,
            "index_var": self.index_var,
            "value_var": self.value_var
        }
        if self.array_vars:
            d["array_vars"] = self.array_vars
        else:
            d["array_values"] = self.array_values
        return d

@dataclass
class AllDifferentConstraint(Constraint):
    def is_satisfied(self, assignment_actual_values: dict[str, int]) -> bool:
        vals = [assignment_actual_values[v] for v in self.scope]
        return len(vals) == len(set(vals))

    def to_normalized_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "allDifferent",
            "scope": self.scope
        }

@dataclass
class CrosswordOverlapConstraint(Constraint):
    pos_i: int
    pos_j: int
    alphabet: list[str] = field(default_factory=list)

    def is_satisfied(self, assignment_actual_values: dict[str, int]) -> bool:
        raise NotImplementedError("Crossword overlap satisfaction requires access to the word dictionaries. Often converted to Table.")

    def to_normalized_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "crosswordOverlap",
            "scope": self.scope,
            "pos_i": self.pos_i,
            "pos_j": self.pos_j,
            "alphabet": self.alphabet
        }

@dataclass
class CSPInstance:
    name: str
    source_path: str
    variables: dict[str, Variable]
    constraints: list[Constraint]
    metadata: dict[str, Any]

    def to_normalized_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_path": self.source_path,
            "variables": [
                {"name": v.name, "domain": v.domain, "kind": v.kind}
                for v in self.variables.values()
            ],
            "constraints": [c.to_normalized_json() for c in self.constraints],
            "metadata": self.metadata
        }
