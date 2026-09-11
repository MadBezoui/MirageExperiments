# =============================================================================
# MIRAGE-R REVISION PROTOCOL -- one-command launcher, 8 parallel jobs (Windows)
# =============================================================================
# Usage (from the repo root, in PowerShell):
#     powershell -ExecutionPolicy Bypass -File scripts\run_revision.ps1
#
# Optional switches:
#     -Fast            120 s / 3 seeds everywhere (smoke-run of the protocol)
#     -SkipSanity      NOT recommended; sanity gate is the P0.1 deliverable
#     -Phase <name>    run a single phase: preflight|sanity|sweep|ablation|
#                      aux|aggregate   (default: all, in order)
#
# The protocol (all phases resumable -- re-run this script after any crash):
#   0. preflight  : python deps, JDK (restored from TRASH if needed), MiniZinc,
#                   Choco jar, feature extraction
#   1. sanity     : toy SAT/UNSAT battery per solver + cross-solver agreement
#                   -- HARD GATE: the sweep will not start if this fails
#   2. sweep      : 8 shard jobs x (7 solvers x 736 instances x 5 seeds, 300 s)
#   3. ablation   : 8 shard jobs x (tau x beta grid, full manifest)
#   4. aux        : memory study, hybrid grid, theory trajectories, GPU
#                   throughput (8 jobs where shardable)
#   5. aggregate  : tables + figures + numbers.tex -> paper/ijoc
# =============================================================================
param(
    [switch]$Fast,
    [switch]$SkipSanity,
    [string]$Phase = "all"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = $Root
$env:PYTHONIOENCODING = "utf-8"

$NumJobs   = 8
$Timeout   = if ($Fast) { 120 } else { 300 }
$Seeds     = if ($Fast) { @(0, 1, 2) } else { @(0, 1, 2, 3, 4) }
$AblTimeout = if ($Fast) { 60 } else { 120 }
$AblSeeds  = if ($Fast) { @(0) } else { @(0, 1, 2) }
$Manifest  = "data\frozen\instances_clean.csv"
$LogDir    = "results\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Info($msg) { Write-Host "[revision] $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "[revision] FATAL: $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- preflight --
function Invoke-Preflight {
    Info "phase 0: preflight"

    # python + deps
    try { $pyv = & python --version 2>&1 } catch { Fail "python not on PATH" }
    Info "  python: $pyv"
    & python -m pip install -q -r requirements-aij.txt
    if ($LASTEXITCODE -ne 0) { Fail "pip install failed" }
    & python -c "import numpy, pandas, matplotlib, scipy, ortools, cpmpy, psutil, torch; print('  deps OK; torch cuda:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { Fail "python dependency import failed" }

    # java: PATH first, then the JDK parked in TRASH
    $java = $null
    try { & java -version 2>&1 | Out-Null; if ($LASTEXITCODE -eq 0) { $java = "java" } } catch {}
    if (-not $java) {
        $trashJava = Join-Path $Root "TRASH\.local\jdk-17.0.10+7\bin\java.exe"
        if (Test-Path $trashJava) { $java = $trashJava; Info "  java: using JDK from TRASH ($trashJava)" }
        else { Fail "no java on PATH and no JDK in TRASH -- install a JRE 17+" }
    } else { Info "  java: on PATH" }
    $env:MIRAGE_JAVA = $java

    # choco fat-jar
    $jar = "baselines\choco\target\choco-baseline-0.1.0.jar"
    if (-not (Test-Path $jar)) { Fail "choco jar missing ($jar) -- rebuild with maven" }
    & $java -cp $jar org.mirager.choco.ChocoBatchRunner 2>&1 | Out-Null  # loads classes
    Info "  choco jar: OK"

    # minizinc (gecode backend)
    $script:HasGecode = $false
    try { & minizinc --version 2>&1 | Out-Null; if ($LASTEXITCODE -eq 0) { $script:HasGecode = $true } } catch {}
    if ($script:HasGecode) { Info "  minizinc: OK (gecode enabled)" }
    else { Write-Host "[revision] WARN: minizinc not on PATH -> gecode rows will be dropped WITH an explicit statement in the paper (allowed by revision P0.1). Install MiniZinc to include gecode." -ForegroundColor Yellow }

    # instance features for the stratified analysis
    & python -m experiments.validation.compute_features --instances-dir data/normalized --out data/frozen/instance_features.csv
    if ($LASTEXITCODE -ne 0) { Fail "feature extraction failed" }
    Info "preflight OK"
}

# ------------------------------------------------------------------- sanity --
function Invoke-Sanity {
    Info "phase 1: sanity gate (P0.1) -- blocks the sweep on failure"
    $solvers = @("mirage", "mirage_regions", "hybrid", "ortools", "choco", "runcsp")
    if ($script:HasGecode) { $solvers += "gecode" }
    & python -m experiments.validation.sanity_check --solvers @solvers --java $env:MIRAGE_JAVA 2>&1 | Tee-Object "$LogDir\sanity.log"
    if ($LASTEXITCODE -ne 0) { Fail "SANITY GATE FAILED -- see $LogDir\sanity.log. The sweep is blocked (P0.1)." }
    Info "sanity gate PASSED"
}

# -------------------------------------------------------------- shard runner --
function Start-Shards([string]$Module, [string[]]$ExtraArgs, [string]$Tag) {
    Info "launching $NumJobs parallel jobs: $Tag"
    $jobs = @()
    for ($i = 0; $i -lt $NumJobs; $i++) {
        $args = @("-m", $Module, "--num-shards", "$NumJobs", "--shard-id", "$i") + $ExtraArgs
        $log = Join-Path $LogDir "$Tag`_shard$i.log"
        $jobs += Start-Process -FilePath "python" -ArgumentList $args `
            -WorkingDirectory $Root -NoNewWindow -PassThru `
            -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    }
    Info "  PIDs: $($jobs.Id -join ', ') -- logs in $LogDir\$Tag`_shard*.log"
    $jobs | Wait-Process
    Info "$($Tag): all $NumJobs jobs finished"
}

# -------------------------------------------------------------------- sweep --
function Invoke-Sweep {
    Info "phase 2: main sweep ($Timeout s, seeds $($Seeds -join ','))"
    $solvers = @("mirage", "mirage_regions", "hybrid", "ortools", "choco", "runcsp")
    if ($script:HasGecode) { $solvers += "gecode" }
    $extra = @("--manifest", $Manifest, "--solvers") + $solvers +
             @("--seeds") + ($Seeds | ForEach-Object { "$_" }) +
             @("--timeout", "$Timeout", "--concurrency", "3",
               "--solver-workers", "1", "--java", "`"$env:MIRAGE_JAVA`"",
               "--out-dir", "results/raw/revision_sweep")
    Start-Shards "experiments.validation.run_sweep" $extra "sweep"
}

# ----------------------------------------------------------------- ablation --
function Invoke-Ablation {
    Info "phase 3: full-set ablation ($AblTimeout s, seeds $($AblSeeds -join ','))"
    $extra = @("--manifest", $Manifest,
               "--seeds") + ($AblSeeds | ForEach-Object { "$_" }) +
             @("--timeout", "$AblTimeout", "--concurrency", "3",
               "--out-dir", "results/raw/revision_ablation")
    Start-Shards "experiments.validation.run_ablation_full" $extra "ablation"
}

# ---------------------------------------------------------------------- aux --
function Invoke-Aux {
    Info "phase 4a: memory study (8 shards, low per-shard concurrency)"
    $extra = @("--manifest", $Manifest, "--timeout", "$AblTimeout",
               "--concurrency", "1", "--out-dir", "results/raw/revision_memory")
    Start-Shards "experiments.validation.run_memory_full" $extra "memory"

    Info "phase 4b: hybrid grid (8 shards)"
    $extra = @("--manifest", $Manifest,
               "--seeds") + ($AblSeeds | ForEach-Object { "$_" }) +
             @("--timeout", "$AblTimeout", "--concurrency", "3",
               "--subset-size", "160",
               "--out-dir", "results/raw/revision_hybrid_grid")
    Start-Shards "experiments.validation.run_hybrid_grid" $extra "hybridgrid"

    Info "phase 4c: theory trajectories (single job)"
    & python -m experiments.validation.run_theory --manifest $Manifest `
        --sample 40 --timeout $AblTimeout --out-dir results/raw/revision_theory `
        2>&1 | Tee-Object "$LogDir\theory.log"

    Info "phase 4d: GPU/batched throughput (single job)"
    & python -m experiments.gpu_throughput --out-dir results/raw/revision_gpu `
        2>&1 | Tee-Object "$LogDir\gpu.log"
}

# ---------------------------------------------------------------- aggregate --
function Invoke-Aggregate {
    Info "phase 5: aggregate -> paper tables/figures/macros"
    & python -m experiments.validation.aggregate `
        --raw-dir results/raw --timeout $Timeout --seeds $Seeds.Count `
        --expected-instances 736 `
        --features data/frozen/instance_features.csv `
        --tables-dir paper/ijoc/tables --figures-dir paper/ijoc/figures `
        2>&1 | Tee-Object "$LogDir\aggregate.log"
    if ($LASTEXITCODE -ne 0) { Fail "aggregation failed" }
    Get-Content "paper\ijoc\tables\soundness.txt"
    Info "done. Compile the paper with:  cd paper\ijoc && pdflatex ms && bibtex ms && pdflatex ms && pdflatex ms"
}

# ---------------------------------------------------------------------- run --
switch ($Phase) {
    "preflight" { Invoke-Preflight }
    "sanity"    { Invoke-Preflight; if (-not $SkipSanity) { Invoke-Sanity } }
    "sweep"     { Invoke-Preflight; Invoke-Sweep }
    "ablation"  { Invoke-Preflight; Invoke-Ablation }
    "aux"       { Invoke-Preflight; Invoke-Aux }
    "aggregate" { Invoke-Aggregate }
    "all" {
        Invoke-Preflight
        if (-not $SkipSanity) { Invoke-Sanity }
        Invoke-Sweep
        Invoke-Aggregate      # early aggregate: paper compiles after phase 2
        Invoke-Ablation
        Invoke-Aux
        Invoke-Aggregate      # final numbers
    }
    default { Fail "unknown -Phase '$Phase'" }
}
