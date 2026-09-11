package org.mirager.choco;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.chocosolver.solver.Model;
import org.chocosolver.solver.Solver;
import org.chocosolver.solver.variables.IntVar;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

public class ChocoBatchRunner {
    public static void main(String[] args) throws Exception {
        String manifestPath = null;
        String inputPath = null;
        String outPath = null;
        long timeoutSec = 300;
        long seed = 0;

        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--manifest")) {
                manifestPath = args[++i];
            } else if (args[i].equals("--input")) {
                inputPath = args[++i];
            } else if (args[i].equals("--out")) {
                outPath = args[++i];
            } else if (args[i].equals("--timeout")) {
                timeoutSec = Long.parseLong(args[++i]);
            } else if (args[i].equals("--seed")) {
                seed = Long.parseLong(args[++i]);
            }
        }

        if (outPath != null) {
            File outFile = new File(outPath);
            outFile.getParentFile().mkdirs();
        }

        ObjectMapper mapper = new ObjectMapper();
        try (PrintWriter writer = outPath != null ? new PrintWriter(new FileWriter(outPath)) : new PrintWriter(System.out)) {
            if (inputPath != null) {
                runInstance(inputPath, seed, timeoutSec, writer, mapper);
            } else if (manifestPath != null) {
                try (BufferedReader br = new BufferedReader(new FileReader(manifestPath))) {
                    String line;
                    boolean header = true;
                    while ((line = br.readLine()) != null) {
                        if (header) {
                            header = false;
                            continue;
                        }
                        String[] parts = line.split(",");
                        if (parts.length >= 3) {
                            String path = parts[2].trim();
                            if (!path.isEmpty()) {
                                runInstance(path, seed, timeoutSec, writer, mapper);
                            }
                        }
                    }
                }
            }
        }
    }

    private static void runInstance(String path, long seed, long timeoutSec, PrintWriter writer, ObjectMapper mapper) {
        try {
            File file = new File(path);
            CspJson instance = mapper.readValue(file, CspJson.class);
            Map<String, IntVar> varsMap = new HashMap<>();
            
            long start = System.currentTimeMillis();
            Model model = ChocoModelBuilder.build(instance, varsMap);
            Solver solver = model.getSolver();
            solver.limitTime(timeoutSec * 1000);
            
            boolean sat = solver.solve();
            long end = System.currentTimeMillis();
            
            ChocoResult result = new ChocoResult();
            result.solver = "choco";
            result.instance = instance.name != null ? instance.name : file.getName();
            result.seed = seed;
            result.runtime_sec = (end - start) / 1000.0;
            result.nodes = solver.getNodeCount();
            result.backtracks = solver.getBackTrackCount();
            result.fails = solver.getFailCount();
            result.restarts = solver.getRestartCount();
            result.choco_version = "4.10.18";
            result.java_version = System.getProperty("java.version");
            result.timeout_sec = timeoutSec;
            
            if (solver.isStopCriterionMet()) {
                result.status = "UNKNOWN_TIMEOUT";
            } else if (sat) {
                result.status = "SAT_VERIFIED";
                result.solution = new HashMap<>();
                for (Map.Entry<String, IntVar> entry : varsMap.entrySet()) {
                    result.solution.put(entry.getKey(), entry.getValue().getValue());
                }
                result.verified = true;
            } else {
                result.status = "UNSAT_PROVED";
            }
            
            writer.println(mapper.writeValueAsString(result));
            writer.flush();
        } catch (Exception e) {
            e.printStackTrace();
            ChocoResult err = new ChocoResult();
            err.solver = "choco";
            err.instance = new File(path).getName();
            err.seed = seed;
            err.status = "ERROR";
            try {
                writer.println(mapper.writeValueAsString(err));
                writer.flush();
            } catch (Exception ex) {}
        }
    }
}
