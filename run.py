"""Command-line entry point for validation, generation, and analysis."""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data_loader import load_dataset
from src.metrics import build_analysis_metrics, calculate_item_metrics, score_results
from src.model_adapters import create_adapter
from src.plots import generate_all_plots
from src.run_experiment import run_experiment
from src.shortlist import rank_vignettes


ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")
    return config


def _project_path(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else ROOT / target


def _load_configured_dataset(config: dict[str, Any], include_unreviewed: bool = False):
    settings = config["dataset"]
    return load_dataset(
        file_path=_project_path(settings["file"]),
        sheet_name=settings["sheet"],
        languages=settings["languages"],
        expected_conditions=settings.get("expected_conditions", config["conditions"]),
        approved_statuses=settings.get("approved_statuses", ["approved"]),
        include_unreviewed=include_unreviewed or settings.get("include_unreviewed", False),
        vignette_ids=settings.get("vignette_ids", []),
    )


def validate_command(config: dict[str, Any]) -> None:
    data = _load_configured_dataset(config)
    if data.empty:
        print(
            "Workbook structure and condition variants are valid, but no rows are "
            "eligible: approve items in the workbook or explicitly enable "
            "dataset.include_unreviewed for development."
        )
        return
    print(
        f"Validated {data['scenario_id'].nunique()} eligible vignette(s), "
        f"{data['language'].nunique()} language(s), {len(data)} prompt variant(s)."
    )


def run_command(config: dict[str, Any]) -> None:
    data = _load_configured_dataset(config)
    if data.empty:
        raise ValueError(
            "No reviewed dataset rows are eligible. Complete review or explicitly "
            "set dataset.include_unreviewed: true for a development run."
        )
    timeout = float(config.get("generation", {}).get("request_timeout_seconds", 90))
    adapters = {
        name: create_adapter(model_config, timeout=timeout)
        for name, model_config in config.get("models", {}).items()
    }
    run_config = dict(config)
    run_config["outputs"] = dict(config["outputs"])
    run_config["outputs"]["results_dir"] = str(
        _project_path(config["outputs"]["results_dir"])
    )
    checkpoint = run_experiment(run_config, data, adapters)
    print(f"Generation checkpoint: {checkpoint}")


def analyze_command(config: dict[str, Any]) -> None:
    results_dir = _project_path(config["outputs"]["results_dir"])
    generations_path = results_dir / "generations.csv"
    if not generations_path.is_file():
        raise FileNotFoundError(f"Generation results not found: {generations_path}")
    generations = pd.read_csv(generations_path, keep_default_na=False)
    data = _load_configured_dataset(config, include_unreviewed=True)
    scores = score_results(generations, data)
    item_metrics = calculate_item_metrics(scores)
    primary = config.get("evaluation", {}).get("primary_contrast", {})
    metrics = build_analysis_metrics(
        item_metrics,
        baseline=primary.get("baseline", "neutral"),
        treatment=primary.get("treatment", "strong_misconception"),
    )
    shortlist_settings = config.get("evaluation", {}).get("shortlist", {})
    shortlist = rank_vignettes(
        metrics,
        target_count=int(shortlist_settings.get("target_count", 12)),
        minimum_valid_responses=int(
            shortlist_settings.get("minimum_valid_responses", 2)
        ),
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(results_dir / "scores.csv", index=False, encoding="utf-8")
    item_metrics.to_csv(results_dir / "item_metrics.csv", index=False, encoding="utf-8")
    metrics.to_csv(results_dir / "vignette_metrics.csv", index=False, encoding="utf-8")
    shortlist.to_csv(results_dir / "vignette_summary.csv", index=False, encoding="utf-8")
    reports_dir = _project_path(config["outputs"].get("reports_dir", "reports/pilot"))
    generate_all_plots(scores, metrics, reports_dir)
    print(f"Scored {len(scores)} generation record(s).")
    print(f"Scores and summaries: {results_dir}")
    print(f"Plots and figure data: {reports_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and analyze the multilingual sycophancy experiment."
    )
    parser.add_argument(
        "command", choices=("validate", "run", "analyze"),
        help="validate workbook, run model generations, or analyze saved results",
    )
    parser.add_argument(
        "--config", default="config/experiment.yaml",
        help="path to the YAML configuration file",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "validate":
        validate_command(config)
    elif args.command == "run":
        run_command(config)
    else:
        analyze_command(config)


if __name__ == "__main__":
    main()