```markdown
# Generative AI Testing Framework

[![CI / MLOps Testing Pipeline](https://github.com/amasresha/generative-ai-testing/actions/workflows/ci.yml/badge.svg)](https://github.com/amasresha/generative-ai-testing/actions/workflows/ci.yml)

This project provides a complete lightweight **testing framework** for validating **Generative AI systems** (text and image generation).

It covers **functional**, **quality**, **bias**, **performance**, and **load** testing — with CI/CD support using GitHub Actions.

---

## 📋 Features

- ✅ **Functional Tests**: Validate text and image output types, placeholder handling, and prompt relevance.
- ✅ **Quality Tests**: Measure ROUGE-1, ROUGE-2, and ROUGE-L scores between outputs and references.
- ✅ **Bias Tests**: Detect gender bias using neutral prompts like "doctor" and "nurse."
- ✅ **Performance Tests**: Measure model latency when generating multiple images.
- ✅ **Load Tests**: Simulate concurrent users posting generation requests (using Locust).

---

## 📂 Project Structure

```plaintext
.
├── Makefile                  # Developer commands (lint, format, test, load test)
├── requirements.txt          # Python dependencies
├── src/
│   └── main.py                # Main text and image generation module
├── tests/
│   ├── conftest.py            # Common test fixtures (sample prompt, outputs)
│   ├── test_functional.py     # Functional testing (types, content, metrics)
│   ├── test_bias.py           # Bias/fairness detection test
│   ├── test_quality.py        # Output quality testing using ROUGE
│   ├── test_performance.py    # Latency testing (timing model inference)
│   ├── test_load.py           # Load testing using Locust
├── dummy_server.py            # FastAPI mock server for load testing
└── .github/
    └── workflows/
        └── ci.yml             # GitHub Actions CI/CD configuration
```

---

## 🚀 How to Run Tests Locally

### 1. Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Format and lint code:

```bash
make format
make lint
```

### 3. Run individual test types:

```bash
make test-functional
make test-quality
make test-bias
make test-performance
make test-load
```

### 4. Run all tests:

```bash
make test-all
```

---

## ⚙️ Continuous Integration (CI/CD)

- **GitHub Actions** automatically runs:
  - Code linting (flake8)
  - Unit/functional/performance/quality/bias tests (pytest)
  - Load testing (Locust)

Every time you **push** or **open a pull request** to the `main` branch.

CI Workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

---

## 📈 Metrics Evaluated

- **ROUGE-1, ROUGE-2, ROUGE-L** (text similarity)
- **Latency (seconds)**
- **Load test request success/failure rates**

---

## 🛠 Technologies Used

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Diffusers](https://huggingface.co/docs/diffusers/)
- [PyTorch](https://pytorch.org/)
- [Pillow (PIL)](https://pillow.readthedocs.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Locust](https://locust.io/)
- [pytest](https://docs.pytest.org/)
- [flake8](https://flake8.pycqa.org/)
- [black](https://black.readthedocs.io/en/stable/)

---

## 🛡️ License

MIT License. Feel free to use, adapt, and contribute.

---
