"""Controlled prompt construction; this module does not score answers."""

from collections.abc import Mapping


def build_prompt(vignette: Mapping, language: str, condition: str) -> str:
    """Build a prompt from one validated language/condition vignette row."""
    row_language = str(vignette.get("language", language)).lower()
    if row_language != language.lower():
        raise ValueError(
            f"Vignette language is {row_language!r}, not requested {language!r}"
        )
    if str(vignette.get("condition", "")).lower() != condition.lower():
        raise ValueError("Vignette row condition does not match requested condition")

    required = ("context", "option_A", "option_B", "closing_question",
                "response_format_instruction")
    missing = [field for field in required if not str(vignette.get(field, "")).strip()]
    if missing:
        raise ValueError(f"Vignette is missing prompt fields: {', '.join(missing)}")

    parts = [
        str(vignette["context"]).strip(),
        f"Option A: {str(vignette['option_A']).strip()}",
        f"Option B: {str(vignette['option_B']).strip()}",
    ]
    stance = str(vignette.get("stance_sentence", "")).strip()
    if condition.lower() == "neutral":
        if stance:
            raise ValueError("Neutral condition must not include a stance sentence")
    elif not stance:
        raise ValueError(f"Condition {condition!r} requires a stance sentence")
    else:
        parts.append(stance)

    parts.extend([
        str(vignette["closing_question"]).strip(),
        str(vignette["response_format_instruction"]).strip(),
    ])
    return "\n\n".join(parts)
