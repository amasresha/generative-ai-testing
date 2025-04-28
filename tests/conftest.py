"""
Common pytest fixtures for generative AI tests.

- Provides `sample_prompt` and `outputs` fixtures for reuse across multiple test files.
- Calls `generate()` once to avoid repeating setup code in every test.
"""

import pytest
from src.main import generate


@pytest.fixture
def sample_prompt():
    return "A happy child playing in a green meadow"


@pytest.fixture
def outputs(sample_prompt):
    return generate(sample_prompt, num_images=2)
