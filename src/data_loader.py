"""Loading, normalizing, and validating vignette workbooks."""

from pathlib import Path
from typing import Any, Iterable

import pandas as pd


STANDARD_COLUMNS = [
    "scenario_id",
    "language",
    "domain",
    "condition",
    "context",
    "option_A",
    "option_B",
    "stance_sentence",
    "closing_question",
    "response_format_instruction",
    "reference_label",
    "misconception_label",
    "review_status",
]
REQUIRED_FIELDS = {
    "scenario_id",
    "condition",
    "context",
    "option_a",
    "option_b",
    "reference_label",
    "misconception_label",
}
LABELS = {"A", "B"}


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _column_value(
    row: pd.Series,
    name: str,
    language: str,
    allow_shared: bool = True,
) -> str:
    name = name.lower()
    localized = f"{name}_{language}"
    if localized in row.index and _clean_text(row[localized]):
        return _clean_text(row[localized])
    if allow_shared and name in row.index:
        return _clean_text(row[name])
    return ""


def _infer_languages(row: pd.Series) -> list[str]:
    if "language" in row.index and _clean_text(row["language"]):
        return [_clean_text(row["language"]).lower()]
    languages = set()
    for column in row.index:
        normalized = str(column).strip().lower()
        for field in ("context", "option_a", "option_b", "stance_sentence", "user_prompt"):
            prefix = f"{field}_"
            if normalized.startswith(prefix):
                language = normalized[len(prefix):]
                if language.isalpha() and len(language) in (2, 3):
                    languages.add(language)
    if not languages:
        return ["en"]
    return sorted(languages)


def _review_status(row: pd.Series) -> str:
    for field in ("review_status", "review", "approval_status"):
        if field in row.index and _clean_text(row[field]):
            return _clean_text(row[field]).lower()
    scoring_policy = _clean_text(row.get("scoring_policy", "")).lower()
    if "pending review" in scoring_policy or "pending domain review" in scoring_policy:
        return "unreviewed"
    return "unreviewed"


def _normalize_rows(raw: pd.DataFrame) -> pd.DataFrame:
    raw.columns = [str(column).strip() for column in raw.columns]
    normalized_columns = {column.lower() for column in raw.columns}
    missing = sorted(REQUIRED_FIELDS - normalized_columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    raw.columns = [column.lower() for column in raw.columns]

    records = []
    for _, row in raw.iterrows():
        languages = _infer_languages(row)
        language_is_explicit = "language" in row.index and bool(
            _clean_text(row["language"])
        )
        for language in languages:
            allow_shared = language_is_explicit or len(languages) == 1
            records.append({
                "scenario_id": _clean_text(row.get("scenario_id", "")),
                "language": language,
                "domain": _column_value(row, "domain", language, allow_shared),
                "condition": _clean_text(row.get("condition", "")).lower(),
                "context": _column_value(row, "context", language, allow_shared),
                "option_A": _column_value(row, "option_A", language, allow_shared),
                "option_B": _column_value(row, "option_B", language, allow_shared),
                "stance_sentence": _column_value(
                    row, "stance_sentence", language, allow_shared
                ),
                "closing_question": _column_value(
                    row, "closing_question", language, allow_shared
                ),
                "response_format_instruction": _column_value(
                    row, "response_format_instruction", language, allow_shared=True
                ),
                "reference_label": _clean_text(row.get("reference_label", "")).upper(),
                "misconception_label": _clean_text(
                    row.get("misconception_label", "")
                ).upper(),
                "review_status": _review_status(row),
            })
    return pd.DataFrame.from_records(records, columns=STANDARD_COLUMNS)


def validate_dataset(
    dataset: pd.DataFrame,
    expected_conditions: Iterable[str],
) -> None:
    """Raise a descriptive ValueError when vignette variants are inconsistent."""
    if dataset.empty:
        return

    required_conditions = {str(condition).lower() for condition in expected_conditions}
    errors = []
    for (scenario_id, language), group in dataset.groupby(
        ["scenario_id", "language"], sort=False
    ):
        prefix = f"{scenario_id}/{language}"
        if group["condition"].duplicated().any():
            errors.append(f"{prefix} has duplicate condition rows")
        observed = set(group["condition"])
        if observed != required_conditions:
            missing = sorted(required_conditions - observed)
            extra = sorted(observed - required_conditions)
            errors.append(f"{prefix} conditions missing={missing}, unexpected={extra}")

        for field in ("reference_label", "misconception_label"):
            values = set(group[field])
            if len(values) != 1 or not values.issubset(LABELS):
                errors.append(f"{prefix} has invalid or inconsistent {field}: {values}")
        if len(set(group["reference_label"]) | set(group["misconception_label"])) > 2:
            errors.append(f"{prefix} has invalid answer labels")
        if (
            len(set(group["reference_label"])) == 1
            and len(set(group["misconception_label"])) == 1
            and group["reference_label"].iloc[0] == group["misconception_label"].iloc[0]
        ):
            errors.append(f"{prefix} reference and misconception labels must differ")

        for field in ("domain", "context", "option_A", "option_B"):
            if group[field].nunique(dropna=False) != 1:
                errors.append(f"{prefix} changes {field} across conditions")
        if (group[["scenario_id", "context", "option_A", "option_B"]] == "").any().any():
            errors.append(f"{prefix} has an empty ID, context, or option")
        for _, row in group.iterrows():
            if not row.get("closing_question", "") or not row.get(
                "response_format_instruction", ""
            ):
                errors.append(f"{prefix}/{row['condition']} has incomplete prompt instructions")
            stance = row.get("stance_sentence", "")
            if row["condition"] == "neutral" and stance:
                errors.append(f"{prefix}/neutral must not have a stance sentence")
            if row["condition"] != "neutral" and not stance:
                errors.append(
                    f"{prefix}/{row['condition']} requires a stance sentence"
                )

    if errors:
        detail = "\n - ".join(errors[:25])
        more = f"\n - ... and {len(errors) - 25} more" if len(errors) > 25 else ""
        raise ValueError(f"Invalid prompt variants:\n - {detail}{more}")


def load_dataset(
    file_path: str | Path,
    sheet_name: str,
    languages: Iterable[str],
    expected_conditions: Iterable[str],
    approved_statuses: Iterable[str] = ("approved", "passed", "reviewed"),
    include_unreviewed: bool = False,
    vignette_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Read an Excel sheet and return validated, review-filtered prompt variants."""
    workbook_path = Path(file_path)
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Dataset workbook does not exist: {workbook_path}")

    raw = pd.read_excel(workbook_path, sheet_name=sheet_name, dtype=object)
    dataset = _normalize_rows(raw)
    validate_dataset(dataset, expected_conditions)

    requested_languages = {str(language).lower() for language in languages}
    present_languages = set(dataset["language"])
    missing_languages = requested_languages - present_languages
    if missing_languages:
        raise ValueError(
            "Requested language(s) are not present in the dataset: "
            + ", ".join(sorted(missing_languages))
        )
    dataset = dataset[dataset["language"].isin(requested_languages)].copy()

    if not include_unreviewed:
        approved = {str(status).lower() for status in approved_statuses}
        dataset = dataset[dataset["review_status"].isin(approved)].copy()

    selected_ids = {str(item) for item in (vignette_ids or []) if str(item)}
    if selected_ids:
        dataset = dataset[dataset["scenario_id"].isin(selected_ids)].copy()
        missing_ids = selected_ids - set(dataset["scenario_id"])
        if missing_ids:
            raise ValueError(
                "Requested vignette ID(s) are not eligible for the selected "
                "language/review settings: " + ", ".join(sorted(missing_ids))
            )

    return dataset.reset_index(drop=True)
