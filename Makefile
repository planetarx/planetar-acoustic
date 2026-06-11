PY ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYBIN := $(VENV)/bin/python

.PHONY: all venv install install-ml install-stream test lint clean run-sai run-classify run-full

all: install

$(VENV)/bin/activate:
	$(PY) -m venv $(VENV)
	$(PIP) install -U pip wheel

venv: $(VENV)/bin/activate

install: venv
	$(PIP) install -e .[dev]

install-ml: venv
	$(PIP) install -e .[dev,ml]

install-stream: venv
	$(PIP) install -e .[dev,stream]

test:
	$(PYBIN) -m pytest

lint:
	$(VENV)/bin/ruff check src tests

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache $(VENV)
	find . -name __pycache__ -type d -exec rm -rf {} +

run-sai:
	$(PYBIN) -m planetar_acoustic sai data/clips/*.wav --out data/sai

run-classify:
	$(PYBIN) -m planetar_acoustic classify data/clips/*.wav

run-full:
	$(PYBIN) -m planetar_acoustic run data/clips/*.wav --broker 127.0.0.1:12001
