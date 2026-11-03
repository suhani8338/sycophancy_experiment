import pandas as pd

from src.model_adapters import GenerationResponse, ModelAdapter
from src.run_experiment import run_experiment


class FakeAdapter(ModelAdapter):
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, **settings):
        self.calls += 1
        return GenerationResponse(
            raw_text="Brief explanation.\nFinal choice: A",
            metadata={"test": True},
        )


def test_successful_jobs_are_checkpointed_and_skipped_on_resume(tmp_path):
    dataset = pd.DataFrame([{
        "scenario_id": "HLT01",
        "language": "en",
        "domain": "Health",
        "condition": "neutral",
        "context": "A scenario.",
        "option_A": "Reference",
        "option_B": "Misconception",
        "stance_sentence": "",
        "closing_question": "Which option is better supported?",
        "response_format_instruction": "End with Final choice: A, B, or UNSURE.",
        "reference_label": "A",
        "misconception_label": "B",
    }])
    config = {
        "experiment": {"name": "test"},
        "dataset": {"languages": ["en"]},
        "conditions": ["neutral"],
        "models": {"fake": {"model": "fake-model"}},
        "repetitions": 1,
        "generation": {"max_retries": 0},
        "outputs": {"results_dir": str(tmp_path)},
    }
    adapter = FakeAdapter()

    checkpoint = run_experiment(config, dataset, {"fake": adapter})
    run_experiment(config, dataset, {"fake": adapter})

    rows = pd.read_csv(checkpoint)
    assert adapter.calls == 1
    assert len(rows) == 1
    assert rows.iloc[0]["status"] == "success"
