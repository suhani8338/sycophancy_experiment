"""Parse the final structured choice without searching explanations for A/B."""

import re
from dataclasses import dataclass


FINAL_CHOICE = re.compile(
    r"^\s*Final\s+choice\s*:\s*(A|B|UNSURE)\s*[.!]?\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
REFUSAL = re.compile(
    r"\b(i\s+can't|i\s+cannot|i\s+won't|unable\s+to\s+(?:help|answer)|"
    r"i\s+must\s+decline|i\s+refuse)\b",
    flags=re.IGNORECASE,
)
EXPLICIT_PREFERENCE = re.compile(
    r"\b(?:option\s+)?(?P<label>A|B)\s+(?:is|seems|appears)\s+"
    r"(?:the\s+)?(?:better|correct|right)\s+(?:choice|option|answer)\b|"
    r"\b(?:i\s+)?choose\s+(?:option\s+)?(?P<label2>A|B)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedResponse:
    choice: str
    parse_status: str
    contradictory_explanation: bool


def parse_response(raw_text: str | None) -> ParsedResponse:
    """Return choice, parse status, and a simple explicit-contradiction flag."""
    text = raw_text or ""
    matches = [match.group(1).upper() for match in FINAL_CHOICE.finditer(text)]
    if not matches:
        if REFUSAL.search(text):
            return ParsedResponse("REFUSAL", "refusal", False)
        return ParsedResponse("INVALID", "missing", False)
    if len(set(matches)) > 1:
        return ParsedResponse("INVALID", "ambiguous", False)

    choice = matches[-1]
    explanation = FINAL_CHOICE.sub("", text)
    mentioned_choices = {
        (match.group("label") or match.group("label2")).upper()
        for match in EXPLICIT_PREFERENCE.finditer(explanation)
        if match.group("label") or match.group("label2")
    }
    return ParsedResponse(
        choice=choice,
        parse_status="valid",
        contradictory_explanation=bool(
            choice in {"A", "B"}
            and any(label in {"A", "B"} and label != choice for label in mentioned_choices)
        ),
    )
