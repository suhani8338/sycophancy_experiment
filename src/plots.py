"""Diagnostic plots with source tables saved alongside each figure."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save_data_and_figure(data: pd.DataFrame, output_dir: Path, stem: str, figure) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / f"{stem}.csv", index=False, encoding="utf-8")
    figure.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_shift_heatmap(metrics: pd.DataFrame, output_dir: str | Path) -> None:
    data = metrics.dropna(subset=["sycophancy_shift"]).copy()
    if data.empty:
        return
    data["panel"] = data["language"].astype(str) + " / " + data["model"].astype(str)
    data["item_label"] = data["scenario_id"].astype(str)
    table = data.pivot_table(
        index="item_label", columns="panel", values="sycophancy_shift", aggfunc="mean"
    )
    figure, axis = plt.subplots(figsize=(max(7, table.shape[1] * 1.4),
                                         max(5, table.shape[0] * 0.25)))
    sns.heatmap(table, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True,
                fmt=".2f", ax=axis)
    axis.set(title="Strong-misconception minus neutral endorsement",
             xlabel="Language / model", ylabel="Vignette")
    _save_data_and_figure(data, Path(output_dir), "sycophancy_shift_heatmap", figure)


def plot_neutral_vs_strong(metrics: pd.DataFrame, output_dir: str | Path) -> None:
    data = metrics.dropna(
        subset=["neutral_endorsement", "strong_misconception_endorsement"]
    ).copy()
    if data.empty:
        return
    data["panel"] = data["language"].astype(str) + " / " + data["model"].astype(str)
    figure, axis = plt.subplots(figsize=(10, max(5, data["scenario_id"].nunique() * 0.3)))
    for panel, group in data.groupby("panel"):
        ordered = group.sort_values("scenario_id")
        for _, row in ordered.iterrows():
            axis.plot(
                [row["neutral_endorsement"], row["strong_misconception_endorsement"]],
                [row["scenario_id"], row["scenario_id"]],
                color="0.75", linewidth=0.8, zorder=1,
            )
        axis.scatter(group["neutral_endorsement"], group["scenario_id"],
                     label=f"{panel}: neutral", marker="o", zorder=2)
        axis.scatter(group["strong_misconception_endorsement"], group["scenario_id"],
                     label=f"{panel}: strong-mis", marker="x", zorder=2)
    axis.set(xlim=(0, 1), xlabel="Misconception endorsement among valid A/B choices",
             ylabel="Vignette", title="Neutral versus strong-misconception endorsement")
    axis.legend(fontsize="small", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_data_and_figure(data, Path(output_dir), "neutral_vs_strong", figure)


def plot_response_distribution(scores: pd.DataFrame, output_dir: str | Path) -> None:
    if scores.empty:
        return
    data = scores.copy()
    data["response_category"] = data["choice"].where(
        data["choice"].isin(["A", "B", "UNSURE", "REFUSAL"]), "INVALID"
    )
    distribution = (
        data.groupby(["language", "model", "condition", "response_category"])
        .size().rename("count").reset_index()
    )
    distribution["proportion"] = distribution["count"] / distribution.groupby(
        ["language", "model", "condition"]
    )["count"].transform("sum")
    distribution["panel"] = (
        distribution["language"].astype(str) + " / " + distribution["model"].astype(str)
    )
    grid = sns.catplot(
        data=distribution,
        x="condition",
        y="proportion",
        hue="response_category",
        col="panel",
        col_wrap=2,
        kind="bar",
        errorbar=None,
        height=4,
        aspect=1.5,
    )
    grid.set(ylim=(0, 1), ylabel="Share of responses", xlabel="Condition")
    grid.set_xticklabels(rotation=25)
    grid.figure.suptitle("Response distribution by condition", y=1.02)
    _save_data_and_figure(
        distribution, Path(output_dir), "response_distribution", grid.figure
    )


def plot_baseline_vs_shift(metrics: pd.DataFrame, output_dir: str | Path) -> None:
    data = metrics.dropna(subset=["neutral_endorsement", "sycophancy_shift"]).copy()
    if data.empty:
        return
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=data, x="neutral_endorsement", y="sycophancy_shift",
        hue="language", style="model", s=70, ax=axis,
    )
    axis.axhline(0, color="0.5", linewidth=0.8)
    axis.set(xlim=(0, 1), ylim=(-1, 1),
             xlabel="Neutral misconception endorsement",
             ylabel="Strong-misconception − neutral endorsement",
             title="Baseline endorsement versus sycophancy shift")
    _save_data_and_figure(data, Path(output_dir), "baseline_vs_shift", figure)


def generate_all_plots(
    scores: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    plot_shift_heatmap(metrics, output_dir)
    plot_neutral_vs_strong(metrics, output_dir)
    plot_response_distribution(scores, output_dir)
    plot_baseline_vs_shift(metrics, output_dir)
