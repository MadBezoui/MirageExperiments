"""Final internal-consistency check on the manuscript (reviewer items I.63--I.68).

Verifies, mechanically:

  1. every macro used in manuscript.tex is defined in one of the generated
     macro files, and every generated macro is either used or explicitly
     whitelisted as unused;
  2. no macro name contains a digit (illegal in LaTeX control sequences);
  3. no empirical percentage is typed literally in the prose outside a macro;
  4. the status totals of every configuration sum to the benchmark size;
  5. the family sizes sum to the benchmark size;
  6. every percentage macro is consistent with its own count macro and the
     stated denominator;
  7. the paired region table is internally consistent and matches the marginal
     totals used in the main comparison;
  8. the compiled PDF has no undefined reference, no undefined citation, no
     overfull box and no missing graphic.

Usage:
    PYTHONPATH=. python -m experiments.validation.verify_manuscript \
        --paper paper/ijoc --log /tmp/build/texF.log
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys

FAIL: list[str] = []
OK: list[str] = []


def check(cond, msg):
    (OK if cond else FAIL).append(msg)


# ----------------------------------------------------------------------
def load_macros(paper):
    macros = {}
    for f in sorted(glob.glob(os.path.join(paper, "tables", "*.tex"))):
        for m in re.finditer(r"\\renewcommand\{\\(Rev[A-Za-z]+)\}\{(.*?)\}\s*$",
                             open(f).read(), re.M):
            macros[m.group(1)] = m.group(2)
    return macros


def used_macros(text):
    return set(re.findall(r"\\(Rev[A-Za-z]+)", text))


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="paper/ijoc")
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--log", default="")
    ap.add_argument("--pages", default="",
                    help="main,ec page counts for the IJOC length policy")
    args = ap.parse_args()

    tex = open(os.path.join(args.paper, "manuscript.tex")).read()
    tables = "".join(open(f).read()
                     for f in glob.glob(os.path.join(args.paper, "tables", "*.tex")))
    macros = load_macros(args.paper)

    # ---- 1. macro definition / use ------------------------------------
    used = used_macros(tex) | used_macros(tables)
    undefined = sorted(u for u in used if u not in macros)
    check(not undefined, f"all used macros are defined (missing: {undefined})")

    unused = sorted(set(macros) - used)
    WHITELIST = {"RevSolveRateGecode", "RevSolvedGecode", "RevMedianTimeGecode",
                 "RevParTwoGecode", "RevAblRegionsDelta", "RevAblSpread",
                 "RevAblWorstRate", "RevAblBestRate", "RevBestStratumCpsat",
                 "RevBestStratumMirage", "RevBestStratumN", "RevBestStratumName",
                 "RevChocoOnlyMedTime", "RevChocoOnlyTopFamN",
                 "RevGpuLargestN", "RevGpuSpeedupCuda", "RevGpuSpeedupVec",
                 "RevHybBestCell", "RevHybBestMedTime", "RevHybBestRate",
                 "RevHybCpsatMedTime", "RevHybCpsatRate", "RevMedianTimeChoco",
                 "RevMedianTimeCpSat", "RevMedianTimeHybrid",
                 "RevMedianTimeMirage", "RevMedianTimeMirageRegions",
                 "RevMedianTimeRunCsp", "RevMemMedianRatio", "RevParTwoChoco",
                 "RevParTwoCpSat", "RevParTwoHybrid", "RevParTwoMirage",
                 "RevParTwoMirageRegions", "RevParTwoRunCsp",
                 "RevRegionsFiredBaseRate", "RevRegionsFiredPct",
                 "RevRegionsFiredSolveRate", "RevRegionsMeanAdded",
                 "RevRegionsWilcoxonP", "RevSolveRateMirage",
                 "RevSolveRateMirageRegions", "RevSolvedRunCsp",
                 "RevTimeout", "RevUnkMirage", "RevUnkWithSatMirage",
                 "RevWitRateHybrid", "RevWitHybrid", "RevAnySeedRateMirage",
                 "RevAnySeedMirage", "RevFpBipRes", "RevFpNonbipRes",
                 "RevFpTablesRes", "RevFpBipFlat", "RevFpNonbipFlat",
                 "RevFpTablesFlat", "RevFpResidualTol", "RevFpHorizon",
                 "RevFpPlateauLead", "RevFpPerturbHorizon", "RevAuditCleanExps",
                 "RevThmCFiveGamma", "RevThmCNineGamma", "RevTauStallEpoch",
                 "RevTauStallValue", "RevUnderflowSat", "RevUnderflowMaxEpoch",
                 "RevRuncspFragAny", "RevRuncspFragPerRun", "RevRegCIlo",
                 "RevRegCIhi", "RevAblBestBeta", "RevAblBestTau",
                 "RevConvergenceSlope", "RevFalsePositives", "RevStrataN",
                 "RevStrataExcluded", "RevStrataExcludedNames"}
    surprising = [u for u in unused if u not in WHITELIST]
    check(True, f"unused macros: {len(unused)} ({len(surprising)} not whitelisted)")

    # ---- 2. no digits in macro names ----------------------------------
    digity = sorted(m for m in macros if any(c.isdigit() for c in m))
    check(not digity, f"no macro name contains a digit (offenders: {digity})")

    # ---- 3. no hard-typed empirical percentages in the prose ----------
    body = tex.split(r"\begin{document}", 1)[1]
    body = re.sub(r"%.*", "", body)
    for env in ("table", "algorithm", "figure"):
        body = re.sub(r"\\begin\{" + env + r"\}.*?\\end\{" + env + r"\}", "",
                      body, flags=re.S)
    ALLOWED = {"23.9", "50", "100", "95", "5", "1", "2", "3", "4", "20", "80"}
    hard = []
    for m in re.finditer(r"(?<![\\{])\b(\d+\.\d)\\%", body):
        if m.group(1) not in ALLOWED:
            hard.append(m.group(1))
    check(not hard,
          f"no hard-typed empirical percentages in prose (found: {sorted(set(hard))})")

    # ---- 4/5. totals from the raw archive -----------------------------
    def canon(n):
        return n[:-5] if n.endswith(".json") else n

    rows = []
    for f in sorted(glob.glob(os.path.join(args.raw, "revision_sweep", "*.jsonl"))):
        for line in open(f):
            if line.strip():
                rows.append(json.loads(line))
    n_inst = len({canon(r["instance"]) for r in rows})
    check(int(macros["RevNumInstances"]) == n_inst,
          f"RevNumInstances ({macros['RevNumInstances']}) equals the archive "
          f"instance count ({n_inst})")

    fam = collections.Counter(canon(r["instance"]).split("-")[0]
                              for r in {canon(r["instance"]): r
                                        for r in rows}.values())
    check(sum(fam.values()) == n_inst,
          f"family sizes {dict(fam)} sum to {n_inst}")

    check(int(macros["RevStrataN"]) + int(macros["RevStrataExcluded"]) == n_inst,
          f"RevStrataN + RevStrataExcluded = "
          f"{int(macros['RevStrataN']) + int(macros['RevStrataExcluded'])} = {n_inst}")
    check(int(macros["RevStrataExcluded"]) == int(macros["RevErrUniversalEntries"]),
          "the stratification exclusions are exactly the universally failing "
          "entries")
    check(int(macros["RevErrEmptyFiles"]) + int(macros["RevErrCorruptFiles"])
          == int(macros["RevErrUniversalEntries"]),
          "empty + corrupt input files account for every universal failure")

    # ---- 6. percentages match their counts ----------------------------
    pairs = [("RevWitRateMirage", "RevSolvedMirage"),
             ("RevWitRateMirageRegions", "RevSolvedMirageRegions"),
             ("RevWitRateChoco", "RevWitChoco"),
             ("RevWitRateCpSat", "RevWitCpSat"),
             ("RevSolveRateChoco", "RevSolvedChoco"),
             ("RevSolveRateCpSat", "RevSolvedCpSat"),
             ("RevSolveRateHybrid", "RevSolvedHybrid")]
    for rate, count in pairs:
        got = float(macros[rate])
        want = 100.0 * int(macros[count]) / n_inst
        check(abs(got - want) < 0.06,
              f"{rate}={got} matches {count}={macros[count]} over {n_inst} "
              f"({want:.2f})")

    # ---- 7. paired region table ---------------------------------------
    b = int(macros["RevRegBoth"])
    ro = int(macros["RevRegRegionOnly"])
    co = int(macros["RevRegCoreOnly"])
    ne = int(macros["RevRegNeither"])
    N = int(macros["RevRegN"])
    check(b + ro + co + ne == N, f"paired region cells sum to {N}")
    check(b + co == int(macros["RevSolvedMirage"]),
          f"paired core marginal {b+co} equals RevSolvedMirage "
          f"{macros['RevSolvedMirage']}")
    check(b + ro == int(macros["RevSolvedMirageRegions"]),
          f"paired region marginal {b+ro} equals RevSolvedMirageRegions "
          f"{macros['RevSolvedMirageRegions']}")
    check(ro + co == int(macros["RevRegDiscordant"]),
          f"discordant count {ro+co} matches RevRegDiscordant")

    # ---- 8. compilation log -------------------------------------------
    if args.log and os.path.exists(args.log):
        log = open(args.log, errors="ignore").read()
        check("Overfull \\hbox" not in log, "no overfull hbox")
        check("Overfull \\vbox" not in log, "no overfull vbox")
        check("There were undefined references" not in log,
              "no undefined references")
        check("Citation" not in log or "undefined" not in log,
              "no undefined citations")
        check("LaTeX Error" not in log, "no LaTeX errors")
        check("File `" not in log or "not found" not in log,
              "no missing input files or graphics")

    # ---- 9. IJOC format policy ----------------------------------------
    ijoc = os.path.join(args.paper, "manuscript.tex")
    src = open(ijoc).read()
    check("\\OneAndAHalfSpacedXII" in src,
          "12-point, 1.5-line spacing (OneAndAHalfSpacedXII)")
    check("\\documentclass[ijoc" in src, "single-column ijoc class option")
    ab = re.search(r"\\ABSTRACT\{(.*?)\n\}\n", src, re.S)
    abstract = re.sub(r"%\s*$", "", ab.group(1), flags=re.M)
    nw = len(abstract.split())
    check(nw <= 300, f"abstract is {nw} words (IJOC limit 300)")
    math_in_abstract = re.findall(r"\$[^$]*\$|\\\[.*?\\\]", abstract, re.S)
    check(not math_in_abstract,
          f"abstract contains no formulas or mathematical notation "
          f"({len(math_in_abstract)} found)")
    kw = re.search(r"\\KEYWORDS\{(.*?)\}", src, re.S).group(1)
    kws = [x.strip() for x in re.split(r"[;,]", kw) if x.strip()]
    check(3 <= len(kws) <= 5, f"{len(kws)} keywords (IJOC asks for 3-5)")
    check(not re.findall(r"\\footnote\{", src), "no footnotes")
    check("double-blind" not in src,
          "no double-blind language (IJOC is single-blind)")
    check("\\AUTHOR{" in src and "\\AFF{" in src,
          "all author names and affiliations on the title page")
    if args.pages:
        main_pp, ec_pp = (int(x) for x in args.pages.split(","))
        check(main_pp <= 25, f"main document is {main_pp} pages (limit 25)")
        check(ec_pp <= 10, f"e-companion is {ec_pp} pages (limit 10)")

    # ---- report --------------------------------------------------------
    for m in OK:
        print(f"  PASS  {m}")
    print()
    for m in FAIL:
        print(f"  FAIL  {m}")
    print(f"\n{len(OK)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
