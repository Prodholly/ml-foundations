# Day 7 -- reproduce everything from one command.
#
#   make all        tests, then benchmark, then experiments
#   make test       test suite only
#   make bench      performance benchmark only
#   make experiment controlled experiments only
#   make clean      remove generated results
#
# Run from the repository root so that pythonpath from the root
# pyproject.toml applies.

PYTHON ?= python
PYTEST ?= pytest
DAY    := 7_prob_stats

.PHONY: all test bench experiment demo clean

all: test bench experiment
	@echo ""
	@echo "=============================================================="
	@echo "Done. Results in $(DAY)/experiments/results/"
	@echo "Figures in      $(DAY)/experiments/results/figures/"
	@echo "=============================================================="

test:
	@echo ">>> test suite"
	PYTHONPATH=$(DAY) $(PYTEST) $(DAY)/tests -v

bench:
	@echo ">>> performance benchmark"
	PYTHONPATH=$(DAY) $(PYTHON) $(DAY)/experiments/run_benchmark.py

experiment:
	@echo ">>> controlled experiments"
	PYTHONPATH=$(DAY) $(PYTHON) $(DAY)/experiments/run_experiment.py

demo:
	PYTHONPATH=$(DAY) $(PYTHON) $(DAY)/examples/run_all.py

clean:
	rm -rf $(DAY)/experiments/results
	find $(DAY) -type d -name __pycache__ -exec rm -rf {} +
	find $(DAY) -type d -name .pytest_cache -exec rm -rf {} +
