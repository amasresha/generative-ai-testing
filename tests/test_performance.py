"""
Performance testing for generative AI outputs.

- Measures latency of the `generate()` function when creating 1 to 3 images.
- Ensures model response times stay within acceptable thresholds.
"""

import time
import pytest
from src.main import generate


@pytest.mark.parametrize("n", [1, 2, 3])
def test_latency(n):
    prompt = "Quick brown fox"
    start = time.time()
    _ = generate(prompt, num_images=n)
    elapsed = time.time() - start
    # expect <5s per image + <1s text
    assert elapsed < 5 * n + 2, f"Too slow: {elapsed:.1f}s for {n} imgs"
