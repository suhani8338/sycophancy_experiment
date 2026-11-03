from src.response_parser import parse_response


def test_parses_final_field_not_mentions_in_explanation():
    parsed = parse_response(
        "Option A is risky, but option B is not supported.\nFinal choice: A"
    )

    assert parsed.choice == "A"
    assert parsed.parse_status == "valid"
    assert not parsed.contradictory_explanation


def test_marks_conflicting_final_fields_ambiguous():
    parsed = parse_response("Final choice: A\nFinal choice: B")

    assert parsed.choice == "INVALID"
    assert parsed.parse_status == "ambiguous"


def test_keeps_refusals_separate_from_invalid_outputs():
    parsed = parse_response("I cannot answer this request.")

    assert parsed.choice == "REFUSAL"
    assert parsed.parse_status == "refusal"


def test_flags_explicit_contradiction():
    parsed = parse_response("Option B is the better choice.\nFinal choice: A")

    assert parsed.contradictory_explanation
