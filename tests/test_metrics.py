import pandas as pd
import pytest

from src.data_loader import validate_dataset
from src.metrics import (
    build_analysis_metrics,
    calculate_item_metrics,
    calculate_sycophancy_shift,
    score_response,
)


def _scored_rows():
    rows = []
    responses = {
        "neutral": ["A", "B"],
        "strong_misconception": ["B", "B"],
        "strong_correct": ["A", "A"],
    }
    for condition, choices in responses.items():
        for repetition, choice in enumerate(choices, start=1):
            rows.append({
                "job_id": f"{condition}-{repetition}",
                "scenario_id": "HLT01",
                "language": "en",
                "model": "test",
                "condition": condition,
                "choice": choice,
                "misconception_endorsed": choice == "B",
                "unsure": False,
                "invalid": False,
                "refusal": False,
                "parse_status": "valid",
                "contradictory_explanation": False,
                "domain": "Health",
            })
    return pd.DataFrame(rows)


def test_scoring_uses_reference_and_misconception_labels():
    vignette = {"reference_label": "B", "misconception_label": "A"}

    score = score_response("Final choice: A", vignette)

    assert score["correct"] is False
    assert score["misconception_endorsed"] is True


def test_primary_shift_and_directional_mirroring():
    item_metrics = calculate_item_metrics(_scored_rows())
    shift = calculate_sycophancy_shift(item_metrics).iloc[0]
    analysis = build_analysis_metrics(item_metrics).iloc[0]

    assert shift["neutral_endorsement"] == pytest.approx(0.5)
    assert shift["strong_misconception_endorsement"] == pytest.approx(1.0)
    assert shift["sycophancy_shift"] == pytest.approx(0.5)
    assert analysis["directional_mirroring"] == pytest.approx(1.0)


def test_metrics_use_latest_checkpoint_for_each_stable_job():
    rows = pd.DataFrame([
        {
            "job_id": "same-job", "scenario_id": "HLT01", "language": "en",
            "model": "test", "condition": "neutral", "choice": "FAILED",
            "misconception_endorsed": pd.NA, "unsure": False, "invalid": False,
            "refusal": False, "parse_status": "generation_failed",
            "contradictory_explanation": False, "generation_failed": True,
            "correct": pd.NA, "domain": "Health",
        },
        {
            "job_id": "same-job", "scenario_id": "HLT01", "language": "en",
            "model": "test", "condition": "neutral", "choice": "A",
            "misconception_endorsed": False, "unsure": False, "invalid": False,
            "refusal": False, "parse_status": "valid",
            "contradictory_explanation": False, "generation_failed": False,
            "correct": True, "domain": "Health",
        },
    ])

    metrics = calculate_item_metrics(rows).iloc[0]

    assert metrics["n_total"] == 1
    assert metrics["n_generation_failed"] == 0
    assert metrics["n_valid"] == 1


def test_dataset_validation_detects_missing_condition():
    rows = []
    for condition in ("neutral", "strong_misconception"):
        rows.append({
            "scenario_id": "HLT01", "language": "en", "condition": condition,
            "domain": "Health", "context": "Context", "option_A": "A",
            "option_B": "B", "reference_label": "A", "misconception_label": "B",
        })

    with pytest.raises(ValueError, match="conditions missing"):
        validate_dataset(pd.DataFrame(rows), ["neutral", "strong_misconception", "strong_correct"])
