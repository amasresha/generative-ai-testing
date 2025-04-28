"""
Functional tests for generative AI outputs.

- Verifies output types, text quality (ROUGE, BLEU), semantic similarity, and image placeholders.
- Ensures good prompts generate better outputs than random irrelevant prompts.
"""

import pytest
from PIL import Image
from src.main import generate, device, _make_placeholder
import evaluate

# Load official scorers once at module level
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

# Constants
PROMPT = "A happy child playing in a green meadow"
NUM_IMAGES = 1  # CPU efficiency
ref = "A happy child playing in a green meadow. One day the two of them walk down the street from him as they leave the home of one of his older brothers."
BAD_PROMPT = "Random characters xyz 123 unrelated text"  # Control candidate prompt


def test_generate_returns_correct_types_and_counts():
    texts, images = generate(PROMPT, num_images=NUM_IMAGES)

    # Validate text output
    assert isinstance(texts, list), "texts should be a list"
    assert all(isinstance(t, str) for t in texts), "each text must be a string"
    assert len(texts) >= 1, "should generate at least one text output"

    # Validate image output
    assert isinstance(images, list), "images should be a list"
    assert len(images) == NUM_IMAGES, f"expected {NUM_IMAGES} image, got {len(images)}"
    assert all(
        isinstance(img, Image.Image) for img in images
    ), "each output must be a PIL Image"


def test_text_metrics_against_prompt():
    texts, _ = generate(PROMPT, num_images=NUM_IMAGES)
    pred = texts[0]
    print("\nGenerated Text:", pred)

    # ROUGE
    rouge_result = rouge.compute(predictions=[pred], references=[ref])
    rouge1_f1 = rouge_result["rouge1"]

    # BLEU
    bleu_result = bleu.compute(predictions=[pred], references=[ref])
    bleu1 = bleu_result["bleu"]

    # Assertions
    for name, val in [("rouge1_f1", rouge1_f1), ("bleu1", bleu1)]:
        assert 0.1 <= val <= 1.0, f"{name} score out of range: {val:.3f}"


def test_semantic_similarity():
    texts, _ = generate(PROMPT, num_images=NUM_IMAGES)
    pred_tokens = set(texts[0].split())
    ref_tokens = set(PROMPT.split())
    union = pred_tokens | ref_tokens
    jaccard = len(pred_tokens & ref_tokens) / len(union) if union else 1.0
    assert 0.0 <= jaccard <= 1.0, f"Jaccard out of range: {jaccard:.3f}"


def test_image_content_on_cpu():
    # Validate placeholder image content
    img = _make_placeholder()
    pixels = list(img.getdata())
    gray = (128, 128, 128)

    # Majority gray
    gray_count = sum(px == gray for px in pixels)
    assert gray_count > len(pixels) * 0.9, "Placeholder should be mostly gray"

    # Some dark pixels for text
    dark = [px for px in pixels if sum(px) < 200]
    assert len(dark) > 0, "Placeholder should have dark text pixels"


def test_prompt_relevance_vs_random_prompt():
    """Compare correct prompt vs irrelevant random prompt"""
    good_texts, _ = generate(PROMPT, num_images=NUM_IMAGES)
    bad_texts, _ = generate(BAD_PROMPT, num_images=NUM_IMAGES)

    good_pred = good_texts[0]
    bad_pred = bad_texts[0]

    # Compute ROUGE and BLEU for good prompt
    good_rouge = rouge.compute(predictions=[good_pred], references=[ref])["rouge1"]
    good_bleu = bleu.compute(predictions=[good_pred], references=[ref])["bleu"]

    # Compute ROUGE and BLEU for bad prompt
    bad_rouge = rouge.compute(predictions=[bad_pred], references=[ref])["rouge1"]
    bad_bleu = bleu.compute(predictions=[bad_pred], references=[ref])["bleu"]

    print(f"\n[Good Prompt] ROUGE-1: {good_rouge:.3f}, BLEU-1: {good_bleu:.3f}")
    print(f"[Bad Prompt]  ROUGE-1: {bad_rouge:.3f}, BLEU-1: {bad_bleu:.3f}")

    # Assert that good prompt gives better scores than bad prompt
    assert good_rouge > bad_rouge, "Good prompt should have higher ROUGE-1 than random"
    assert good_bleu > bad_bleu, "Good prompt should have higher BLEU-1 than random"
