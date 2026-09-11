package org.mirager.choco;

import java.util.Map;

public class ChocoResult {
    public String solver;
    public String instance;
    public long seed;
    public String status; // SAT_VERIFIED | UNSAT_PROVED | UNKNOWN_TIMEOUT | ERROR
    public double runtime_sec;
    public long nodes;
    public long backtracks;
    public long fails;
    public long restarts;
    public Map<String, Integer> solution;
    public boolean verified;
    public String choco_version;
    public String java_version;
    public double timeout_sec;
}
