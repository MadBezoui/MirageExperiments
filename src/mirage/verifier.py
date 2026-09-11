from src.mirage.csp_core import CSPInstance

def verify_assignment(instance: CSPInstance, assignment: dict[str, int]) -> bool:
    """Returns True if the assignment satisfies all constraints in the instance."""
    for constr in instance.constraints:
        if not constr.is_satisfied(assignment):
            return False
    return True

def count_violations(instance: CSPInstance, assignment: dict[str, int]) -> int:
    """Returns the number of violated constraints."""
    count = 0
    for constr in instance.constraints:
        if not constr.is_satisfied(assignment):
            count += 1
    return count

def violated_constraints(instance: CSPInstance, assignment: dict[str, int]) -> list[str]:
    """Returns the IDs of the violated constraints."""
    violated = []
    for constr in instance.constraints:
        if not constr.is_satisfied(assignment):
            violated.append(constr.id)
    return violated
