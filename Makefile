PY=python
PIP=$(PY) -m pip

.PHONY: init lint format test-functional test-performance test-bias test-quality test-load test-all clean

init:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install black

lint:
	black src tests
	flake8 src tests

format:
	black src tests

test-functional:
	PYTHONPATH=$(PWD) pytest tests/test_functional.py -q -s

test-performance:
	PYTHONPATH=$(PWD) pytest tests/test_performance.py -q -s

test-bias:
	PYTHONPATH=$(PWD) pytest tests/test_bias.py -q -s

test-quality:
	PYTHONPATH=$(PWD) pytest tests/test_quality.py -q -s

test-load:
	PYTHONPATH=$(PWD) locust --headless -u 10 -r 2 -t 15s -f tests/test_load.py

test-all:
	PYTHONPATH=. pytest --maxfail=1 --disable-warnings -q 2> /dev/null

clean:
	rm -rf .venv __pycache__
