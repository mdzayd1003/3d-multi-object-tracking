.PHONY: test quick crosscheck clean

test:
	python -m pytest -q

quick:
	python -m mot --all

crosscheck:
	python scripts/make_reference.py
	node scripts/crosscheck.mjs

clean:
	rm -rf .pytest_cache __pycache__ mot/__pycache__ tests/__pycache__
