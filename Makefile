test:
	pytest tests/

choco-build:
	cd baselines/choco && mvn clean package

normalize:
	python scripts/normalize_benchmarks.py --input benchmarks --output data/normalized --report results/processed/parser_census.csv

smoke:
	pytest tests/test_mirage_smoke.py

pilot:
	python experiments/run_batch.py --manifest data/frozen/instances_clean.csv --solvers fourier crigs mirage_full choco --seeds 0 --timeout 30 --out results/raw/pilot.jsonl --dump-traces

sweep:
	python experiments/run_batch.py --manifest data/frozen/instances_clean.csv --solvers fourier crigs mirage_full choco --seeds 0 1 2 3 4 --timeout 300 --out results/raw/main_sweep.jsonl --dump-traces

unsat:
	python experiments/run_batch.py --manifest data/frozen/instances_unsat.csv --solvers mirage_full choco --seeds 0 1 2 3 4 --timeout 300 --out results/raw/unsat_sweep.jsonl --dump-traces

figures:
	python experiments/analyze_results.py --input results/raw/main_sweep.jsonl --outdir results/processed/main

pipeline:
	python experiments/run_pipeline.py --input benchmarks --normalized-dir data/normalized --solvers fourier crigs mirage_full choco --seeds 0 1 2 3 4 --timeout 300 --out results/raw/pipeline_sweep.jsonl --dump-traces

paper:
	cd paper && pdflatex main.tex

clean-results:
	rm -rf results/raw/* results/processed/* results/figures/*

# ---- IJOC submission targets ----
# `make verify` is the one to run: it regenerates everything the paper
# reports and requires each regenerated table to come back byte-identical.
setup:
	pip install -r requirements.txt

verify:
	./scripts/reproduce.sh verify

reproduce:
	./scripts/reproduce.sh all

theory:
	./scripts/reproduce.sh theory

diagnostics:
	./scripts/reproduce.sh diagnostics

audit:
	./scripts/reproduce.sh audit

submission:
	python3 scripts/build_submission.py

# ---- weekend validation ----
weekend-launch:
	bash scripts/launch_all_weekend.sh

weekend-shard:
	bash scripts/run_weekend_validation.sh $(SHARD) $(or $(NUM_SHARDS),5)

weekend-aggregate:
	bash scripts/aggregate_weekend.sh $(or $(TIMEOUT),300)

