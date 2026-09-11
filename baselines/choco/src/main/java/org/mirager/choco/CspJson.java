package org.mirager.choco;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;
import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public class CspJson {
    public String name;
    public String source_path;
    public List<Variable> variables;
    public List<Constraint> constraints;
    public Map<String, Object> metadata;

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Variable {
        public String name;
        public List<Integer> domain;
        public String kind;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Constraint {
        public String id;
        public String type; // "table", "sum", "count", "element", "allDifferent"
        public List<String> scope;
        
        // for table
        public Boolean positive;
        public List<List<Integer>> tuples;
        
        // for sum
        public List<Integer> coeffs;
        public String op; 
        public Integer rhs;
        
        // for count
        public Integer value;
        public List<Integer> values;
        public String operator;
        
        // for element
        public List<Integer> array_values; 
        public String index_var;
        public String value_var;
    }
}
