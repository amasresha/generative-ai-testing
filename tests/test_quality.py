"""
Quality testing for generative AI text outputs.

- Measures ROUGE-1, ROUGE-2, and ROUGE-L scores between generated text and references.
- Ensures model outputs maintain a minimum similarity to expected human-like answers.
"""

import evaluate
import pytest


@pytest.fixture(scope="session")
def rouge():
    return evaluate.load("rouge")


@pytest.fixture(scope="session")
def sample_data():
    """Provide prompts and their expected reference outputs."""
    return [
        (
            "A happy child playing in a green meadow",
            "A child happily playing in a lush green field.",
        ),
        ("A cat sleeping on a sofa", "A cat taking a nap on a cozy sofa."),
        (
            "A futuristic city at night",
            "A glowing futuristic city skyline under the night sky.",
        ),
    ]


@pytest.mark.parametrize(
    "prompt, reference",
    [
        (
            "A happy child playing in a green meadow",
            "A child happily playing in a lush green field.",
        ),
        ("A cat sleeping on a sofa", "A cat taking a nap on a cozy sofa."),
        (
            "A futuristic city at night",
            "A glowing futuristic city skyline under the night sky.",
        ),
    ],
)
def test_text_quality(prompt, reference, rouge):
    from src.main import generate

    texts, _ = generate(prompt, num_images=1)
    pred = texts[0]
    print(f"\nPrompt: {prompt}")
    print(f"Model Output: {pred}")
    print(f"Reference: {reference}")

    scores = rouge.compute(predictions=[pred], references=[reference])

    rouge1_score = scores["rouge1"]
    rouge2_score = scores["rouge2"]
    rougel_score = scores["rougeL"]

    print(
        f"ROUGE-1: {rouge1_score:.3f} | "
        f"ROUGE-2: {rouge2_score:.3f} | "
        f"ROUGE-L: {rougel_score:.3f}"
    )

    assert 0.1 <= rouge1_score <= 1.0, f"ROUGE-1 too low: {rouge1_score:.3f}"
    assert 0.035 <= rouge2_score <= 1.0, f"ROUGE-2 too low: {rouge2_score:.3f}"
    assert 0.1 <= rougel_score <= 1.0, f"ROUGE-L too low: {rougel_score:.3f}"
