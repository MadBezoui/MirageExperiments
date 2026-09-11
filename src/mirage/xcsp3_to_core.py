import sys
import os

# Add the project root to the path so we can import xcsp3_parser
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from xcsp3_parser import parse_xcsp3, Instance as OldInstance, Constraint as OldConstraint
from src.mirage.csp_core import (
    CSPInstance, Variable, TableConstraint, SumConstraint,
    CountConstraint, ElementConstraint, AllDifferentConstraint,
    CrosswordOverlapConstraint
)

def convert_to_core(old_inst: OldInstance, status: str, unsupported_reason: str, relaxed: bool) -> CSPInstance:
    variables = {}
    for i, name in enumerate(old_inst.var_names):
        domain = old_inst.domains[i]
        variables[name] = Variable(name=name, domain=domain, kind="int")

    constraints = []
    for c in old_inst.constraints:
        scope_names = [old_inst.var_names[i] for i in c.scope]
        
        if c.semantics in ("supports", "conflicts"):
            constraints.append(TableConstraint(
                id=c.name,
                scope=scope_names,
                positive=(c.semantics == "supports"),
                tuples=list(c.tuples)
            ))
        elif c.semantics == "sum":
            # old sum tuples is a dict: {'list': [...], 'coeffs': [...], 'op': 'eq', 'operand': ..., 'operand_type': 'int'/'var'}
            d = c.tuples
            scope = [old_inst.var_names[i] for i in d["list"]]
            coeffs = list(d["coeffs"])
            # Handling operand_type == "var" by moving it to LHS
            op = d["op"]
            if d["operand_type"] == "var":
                rhs_var = old_inst.var_names[d["operand"]]
                scope.append(rhs_var)
                coeffs.append(-1)
                rhs = 0
            else:
                rhs = d["operand"]
            constraints.append(SumConstraint(
                id=c.name, scope=scope, coeffs=coeffs, operator=op, rhs=rhs
            ))
        elif c.semantics == "count":
            d = c.tuples
            scope = [old_inst.var_names[i] for i in d["list"]]
            # operand_type var not natively supported in our count constraint dataclass without adding it to the RHS explicitly,
            # but wait, the new dataclass assumes rhs is an int.
            if d["operand_type"] == "var":
                status = "unsupported"
                unsupported_reason = f"count with var RHS not supported: {c.name}"
                break
            constraints.append(CountConstraint(
                id=c.name, scope=scope, values=[d["value"]], operator=d["op"], rhs=d["operand"]
            ))
        elif c.semantics == "element":
            d = c.tuples
            index_var = old_inst.var_names[d["index"]] if d["index_type"] == "var" else None
            if index_var is None:
                status = "unsupported"
                unsupported_reason = f"element with int index not supported: {c.name}"
                break
            
            array_vars = [old_inst.var_names[i] for i in d["list"]]
            if d["operand_type"] == "var":
                value_var = old_inst.var_names[d["operand"]]
            else:
                # We need a value_var, element constraint in new core assumes value_var is a string.
                # If operand is int, this is unsupported in the current strict MIRAGE element constraint.
                status = "unsupported"
                unsupported_reason = f"element with int value not supported: {c.name}"
                break
            
            # The scope must include index_var, array_vars, value_var
            scope = [index_var, value_var] + array_vars
            constraints.append(ElementConstraint(
                id=c.name, scope=scope, index_var=index_var, value_var=value_var, array_vars=array_vars
            ))
        elif c.semantics == "alldiff":
            constraints.append(AllDifferentConstraint(
                id=c.name, scope=scope_names
            ))
        else:
            status = "unsupported"
            unsupported_reason = f"Unsupported semantics in convert: {c.semantics}"
            break

    metadata = {
        "family": old_inst.family,
        "parser_status": status,
        "unsupported_reason": unsupported_reason,
        "relaxed": relaxed
    }

    return CSPInstance(
        name=old_inst.name,
        source_path=old_inst.path,
        variables=variables,
        constraints=constraints,
        metadata=metadata
    )

def parse_and_convert(xml_path: str) -> CSPInstance:
    import xcsp3_parser
    # Force strict parsing first
    xcsp3_parser.STRICT_XCSP3 = True
    
    status = "clean"
    unsupported_reason = None
    relaxed = False
    old_inst = None
    
    try:
        old_inst = parse_xcsp3(xml_path, max_expansion=50000)
    except ValueError as e:
        if "Optimization objectives not supported" in str(e) or "Unsupported constraint" in str(e) or "expansion" in str(e):
            # Try lenient mode
            xcsp3_parser.STRICT_XCSP3 = False
            try:
                old_inst = parse_xcsp3(xml_path, max_expansion=50000)
                relaxed = True
                status = "relaxed"
                unsupported_reason = str(e)
            except Exception as e2:
                status = "parse_error"
                unsupported_reason = str(e2)
        else:
            status = "parse_error"
            unsupported_reason = str(e)
    except Exception as e:
        status = "parse_error"
        unsupported_reason = str(e)

    if old_inst is None:
        # Create a dummy instance just to carry the metadata
        return CSPInstance(
            name=os.path.basename(xml_path),
            source_path=xml_path,
            variables={},
            constraints=[],
            metadata={
                "family": "unknown",
                "parser_status": status,
                "unsupported_reason": unsupported_reason,
                "relaxed": relaxed
            }
        )
        
    return convert_to_core(old_inst, status, unsupported_reason, relaxed)
