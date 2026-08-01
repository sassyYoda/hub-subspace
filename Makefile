.PHONY: venv install test clean-cache

venv:
	uv venv --python 3.11 .venv

install:
	uv pip install --python .venv/bin/python -e .

test:
	.venv/bin/python -m pytest -q

clean-cache:
	rm -rf results/cache/
