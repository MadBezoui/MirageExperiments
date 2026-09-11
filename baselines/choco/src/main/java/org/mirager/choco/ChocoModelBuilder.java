package org.mirager.choco;

import org.chocosolver.solver.Model;
import org.chocosolver.solver.variables.IntVar;
import org.chocosolver.solver.constraints.extension.Tuples;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ChocoModelBuilder {

    public static Model build(CspJson instance, Map<String, IntVar> varsMap) {
        Model model = new Model(instance.name != null ? instance.name : "CSP");

        if (instance.variables != null) {
            for (CspJson.Variable v : instance.variables) {
                int[] dom = v.domain.stream().mapToInt(i -> i).toArray();
                IntVar chocoVar = model.intVar(v.name, dom);
                varsMap.put(v.name, chocoVar);
            }
        }

        if (instance.constraints != null) {
            for (CspJson.Constraint c : instance.constraints) {
                IntVar[] scope = new IntVar[0];
                if (c.scope != null) {
                    scope = c.scope.stream().map(varsMap::get).toArray(IntVar[]::new);
                }

                switch (c.type) {
                    case "table":
                        Tuples tuples = new Tuples(c.positive != null ? c.positive : true);
                        for (List<Integer> t : c.tuples) {
                            tuples.add(t.stream().mapToInt(i -> i).toArray());
                        }
                        model.table(scope, tuples).post();
                        break;
                    case "sum":
                        int[] coeffs = c.coeffs.stream().mapToInt(i -> i).toArray();
                        String sumOp = c.operator != null ? c.operator : c.op;
                        if ("eq".equals(sumOp) || "==".equals(sumOp)) sumOp = "=";
                        else if ("le".equals(sumOp)) sumOp = "<=";
                        else if ("ge".equals(sumOp)) sumOp = ">=";
                        else if ("lt".equals(sumOp)) sumOp = "<";
                        else if ("gt".equals(sumOp)) sumOp = ">";
                        else if ("ne".equals(sumOp)) sumOp = "!=";
                        model.scalar(scope, coeffs, sumOp, c.rhs).post();
                        break;
                    case "count":
                        IntVar limit = model.intVar("limit_" + c.id, 0, scope.length);
                        int countValue = (c.values != null && !c.values.isEmpty()) ? c.values.get(0) : (c.value != null ? c.value : 0);
                        model.count(countValue, scope, limit).post();
                        String countOp = c.operator != null ? c.operator : c.op;
                        if ("eq".equals(countOp) || "==".equals(countOp)) countOp = "=";
                        else if ("le".equals(countOp)) countOp = "<=";
                        else if ("ge".equals(countOp)) countOp = ">=";
                        else if ("lt".equals(countOp)) countOp = "<";
                        else if ("gt".equals(countOp)) countOp = ">";
                        else if ("ne".equals(countOp)) countOp = "!=";
                        model.arithm(limit, countOp, c.rhs).post();
                        break;
                    case "element":
                        int[] array = c.array_values.stream().mapToInt(i -> i).toArray();
                        IntVar indexVar = varsMap.get(c.index_var != null ? c.index_var : c.scope.get(0));
                        IntVar valueVar = varsMap.get(c.value_var != null ? c.value_var : c.scope.get(1));
                        model.element(valueVar, array, indexVar, 0).post();
                        break;
                    case "allDifferent":
                        model.allDifferent(scope).post();
                        break;
                    default:
                        System.err.println("Unsupported constraint type: " + c.type);
                }
            }
        }
        return model;
    }
}
