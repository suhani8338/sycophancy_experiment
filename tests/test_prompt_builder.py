import pytest

from src.prompt_builder import build_prompt


def _vignette(condition="strong_misconception", stance="I am certain option B is right."):
    return {
        "language": "en",
        "condition": condition,
        "context": "A short scenario.",
        "option_A": "Use the reference action.",
        "option_B": "Use the misconception action.",
        "stance_sentence": stance,
        "closing_question": "Which option is better supported?",
        "response_format_instruction": "End with Final choice: A, B, or UNSURE.",
    }


def test_prompt_keeps_options_and_inserts_stance():
    prompt = build_prompt(_vignette(), "en", "strong_misconception")

    assert "Option A: Use the reference action." in prompt
    assert "Option B: Use the misconception action." in prompt
    assert "I am certain option B is right." in prompt
    assert prompt.endswith("End with Final choice: A, B, or UNSURE.")


def test_neutral_prompt_rejects_stance():
    with pytest.raises(ValueError, match="must not include"):
        build_prompt(_vignette("neutral"), "en", "neutral")


def test_non_neutral_prompt_requires_stance():
    with pytest.raises(ValueError, match="requires a stance"):
        build_prompt(_vignette(stance=""), "en", "strong_misconception")
