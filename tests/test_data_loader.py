import pandas as pd

from src.data_loader import _normalize_rows


def test_wide_multilingual_rows_do_not_reuse_shared_context_as_translation():
    raw = pd.DataFrame([{
        "scenario_id": "HLT01",
        "condition": "neutral",
        "context": "English-only context",
        "option_A": "English option A",
        "option_B": "English option B",
        "reference_label": "A",
        "misconception_label": "B",
        "user_prompt_en": "English prompt",
        "user_prompt_hi": "Hindi prompt",
        "response_format_instruction": "Final choice: A, B, or UNSURE.",
    }])

    normalized = _normalize_rows(raw).set_index("language")

    assert normalized.loc["en", "context"] == ""
    assert normalized.loc["hi", "context"] == ""
    assert normalized.loc["en", "response_format_instruction"] == (
        "Final choice: A, B, or UNSURE."
    )
