"""Job creation, retry handling, and resumable generation checkpoints."""

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.model_adapters import AdapterError, GenerationResponse, ModelAdapter
from src.prompt_builder import build_prompt


@dataclass(frozen=True)
class ExperimentJob:
    job_id: str
    experiment: str
    scenario_id: str
    language: str
    condition: str
    model: str
    model_id: str
    repetition: int
    vignette: dict[str, Any]


def _stable_id(parts: list[str]) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def create_jobs(
    dataset: pd.DataFrame,
    languages: list[str],
    conditions: list[str],
    models: Mapping[str, Any],
    repetitions: int,
    experiment_name: str = "experiment",
    generation_settings: Mapping[str, Any] | None = None,
) -> list[ExperimentJob]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    jobs = []
    for _, row in dataset.iterrows():
        language = str(row["language"])
        condition = str(row["condition"])
        if language not in languages or condition not in conditions:
            continue
        vignette = row.to_dict()
        prompt = build_prompt(vignette, language, condition)
        for model_name, model_config in models.items():
            model_id = str(model_config.get("model", model_name))
            provider_identity = {
                key: model_config[key]
                for key in ("provider", "endpoint", "api_version")
                if key in model_config
            }
            for repetition in range(1, repetitions + 1):
                job_id = _stable_id([
                    experiment_name,
                    str(row["scenario_id"]),
                    language,
                    condition,
                    str(model_name),
                    model_id,
                    json.dumps(provider_identity, sort_keys=True),
                    json.dumps(
                        {
                            key: value
                            for key, value in (generation_settings or {}).items()
                            if key not in {
                                "max_retries",
                                "retry_backoff_seconds",
                                "request_timeout_seconds",
                            }
                        },
                        sort_keys=True,
                    ),
                    str(repetition),
                    prompt,
                ])
                jobs.append(ExperimentJob(
                    job_id=job_id,
                    experiment=experiment_name,
                    scenario_id=str(row["scenario_id"]),
                    language=language,
                    condition=condition,
                    model=str(model_name),
                    model_id=model_id,
                    repetition=repetition,
                    vignette=vignette,
                ))
    return jobs


def _completed_job_ids(checkpoint_path: Path) -> set[str]:
    if not checkpoint_path.is_file():
        return set()
    existing = pd.read_csv(checkpoint_path, usecols=["job_id", "status"], dtype=str)
    return set(existing.loc[existing["status"] == "success", "job_id"])


def _checkpoint(
    checkpoint_path: Path,
    row: dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        checkpoint_path,
        mode="a",
        header=not checkpoint_path.exists(),
        index=False,
        encoding="utf-8",
    )


def generate_with_retry(
    adapter: ModelAdapter,
    prompt: str,
    settings: dict[str, Any],
) -> GenerationResponse:
    max_retries = int(settings.get("max_retries", 0))
    backoff = float(settings.get("retry_backoff_seconds", 1))
    for attempt in range(max_retries + 1):
        try:
            return adapter.generate(prompt, **settings)
        except AdapterError:
            if attempt >= max_retries:
                raise
            time.sleep(backoff * (2 ** attempt))
    raise RuntimeError("Generation retry loop ended unexpectedly")


def run_experiment(
    config: dict[str, Any],
    dataset: pd.DataFrame,
    model_adapters: Mapping[str, ModelAdapter],
) -> Path:
    """Run configured jobs, appending one durable checkpoint row per attempt."""
    model_configs = config.get("models", {})
    if not model_configs:
        raise ValueError("No models configured; add at least one entry under 'models'")
    missing_adapters = set(model_configs) - set(model_adapters)
    if missing_adapters:
        raise ValueError(
            "Missing adapters for configured models: " + ", ".join(sorted(missing_adapters))
        )
    if dataset.empty:
        raise ValueError("No eligible dataset rows are available to run")

    experiment = config.get("experiment", {}).get("name", "experiment")
    dataset_config = config["dataset"]
    jobs = create_jobs(
        dataset=dataset,
        languages=dataset_config["languages"],
        conditions=config["conditions"],
        models=model_configs,
        repetitions=int(config["repetitions"]),
        experiment_name=experiment,
        generation_settings=config.get("generation", {}),
    )
    checkpoint = Path(config["outputs"]["results_dir"]) / "generations.csv"
    completed = _completed_job_ids(checkpoint)
    settings = dict(config.get("generation", {}))
    total = sum(job.job_id not in completed for job in jobs)

    for index, job in enumerate((job for job in jobs if job.job_id not in completed), 1):
        prompt = build_prompt(job.vignette, job.language, job.condition)
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "job_id": job.job_id,
            "experiment": job.experiment,
            "scenario_id": job.scenario_id,
            "language": job.language,
            "condition": job.condition,
            "model": job.model,
            "model_id": job.model_id,
            "repetition": job.repetition,
            "timestamp_utc": now,
            "prompt": prompt,
            "raw_text": "",
            "metadata_json": "{}",
            "status": "success",
            "error": "",
        }
        try:
            response = generate_with_retry(model_adapters[job.model], prompt, settings)
            record["raw_text"] = response.raw_text
            record["metadata_json"] = json.dumps(response.metadata, ensure_ascii=False)
        except AdapterError as error:
            record["status"] = "failed"
            record["error"] = str(error)
        _checkpoint(checkpoint, record)
        print(f"[{index}/{total}] {job.job_id} {record['status']}")

    return checkpoint
