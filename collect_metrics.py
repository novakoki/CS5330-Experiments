"""
Parse MMDetection3D experiment logs under outputs/mmdet, aggregate metrics, and
draw comparison plots for different approaches.

Outputs:
- outputs/mmdet/experiment_metrics.csv: best metrics for every run that has evals
- outputs/mmdet/analysis/best_by_experiment.csv: best run per experiment
- outputs/mmdet/analysis/training_loss.html: mean training loss per epoch
- outputs/mmdet/analysis/val_metrics.html: validation car mAP/NDS per epoch
- outputs/mmdet/analysis/best_metric_comparison.html: bar chart of best car mAP/NDS
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "outputs" / "mmdet"
ANALYSIS_DIR = OUTPUT_ROOT / "analysis"
CSV_OUTPUT = OUTPUT_ROOT / "experiment_metrics.csv"
BEST_BY_EXPERIMENT = ANALYSIS_DIR / "best_by_experiment.csv"
TRAIN_CURVE_HTML = ANALYSIS_DIR / "training_loss.html"
VAL_CURVE_HTML = ANALYSIS_DIR / "val_metrics.html"
BEST_BAR_HTML = ANALYSIS_DIR / "best_metric_comparison.html"

FLOAT_PATTERN = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
TRAIN_RE = re.compile(
    rf"Epoch\(train\)\s+\[(\d+)\]\[\s*(\d+)/(\d+)\].*?loss:\s*{FLOAT_PATTERN}"
)
VAL_EPOCH_RE = re.compile(r"Epoch\(val\)\s+\[(\d+)\]")
NDS_RE = re.compile(r"NDS:\s*" + FLOAT_PATTERN)
MAP_RE = re.compile(r"mAP:\s*" + FLOAT_PATTERN)
CAR_AP_RE = re.compile(r"car_AP_dist_(0\.5|1\.0|2\.0|4\.0):\s*" + FLOAT_PATTERN)


@dataclass
class ParsedLog:
    experiment: str
    run_id: str
    run_suffix: str
    log_path: Path
    train_rows: List[dict]
    val_rows: List[dict]


def iter_log_files() -> Iterable[Tuple[str, str, Path]]:
    """Yield (experiment, run_id, log_path) tuples for every .log file."""
    for log_path in sorted(OUTPUT_ROOT.rglob("*.log")):
        rel = log_path.relative_to(OUTPUT_ROOT)
        experiment = rel.parts[0]
        run_id = log_path.parent.relative_to(OUTPUT_ROOT).as_posix()
        yield experiment, run_id, log_path


def parse_log(experiment: str, run_id: str, log_path: Path) -> ParsedLog:
    """Parse a single text log to extract training loss and validation metrics."""
    train_rows: List[dict] = []
    val_rows: List[dict] = []
    run_suffix = log_path.parent.name

    with log_path.open() as f:
        for line in f:
            if "Epoch(train)" in line:
                m = TRAIN_RE.search(line)
                if not m:
                    continue
                epoch = int(m.group(1))
                iter_idx = int(m.group(2))
                loss = float(m.group(4))
                train_rows.append(
                    {
                        "experiment": experiment,
                        "run_id": run_id,
                        "run_suffix": run_suffix,
                        "epoch": epoch,
                        "iter_in_epoch": iter_idx,
                        "loss": loss,
                        "log_path": str(log_path),
                    }
                )
            elif "Epoch(val)" in line:
                epoch_match = VAL_EPOCH_RE.search(line)
                if not epoch_match:
                    continue
                epoch = int(epoch_match.group(1))
                nds_match = NDS_RE.search(line)
                map_match = MAP_RE.search(line)
                nds = float(nds_match.group(1)) if nds_match else None
                map_ = float(map_match.group(1)) if map_match else None
                car_ap_vals = [float(m.group(2)) for m in CAR_AP_RE.finditer(line)]
                car_map = sum(car_ap_vals) / len(car_ap_vals) if car_ap_vals else None
                if nds is None and map_ is None and car_map is None:
                    continue
                val_rows.append(
                    {
                        "experiment": experiment,
                        "run_id": run_id,
                        "run_suffix": run_suffix,
                        "epoch": epoch,
                        "car_mAP": car_map,
                        "NDS": nds,
                        "mAP": map_,
                        "log_path": str(log_path),
                    }
                )
    return ParsedLog(experiment, run_id, run_suffix, log_path, train_rows, val_rows)


def choose_best_row(group: pd.DataFrame) -> pd.Series | None:
    """Pick the row with the highest car mAP, falling back to NDS then mAP."""
    if group.empty:
        return None
    for col in ("car_mAP", "NDS", "mAP"):
        if col in group.columns and group[col].notna().any():
            idx = group[col].idxmax()
            return group.loc[idx]
    return None


def pick_best_runs(best_per_run: pd.DataFrame) -> pd.DataFrame:
    """Select the strongest run per experiment."""
    best_rows = []
    for exp, group in best_per_run.groupby("experiment"):
        if group.empty:
            continue
        sorted_group = group.sort_values(
            by=["best_car_mAP", "best_NDS", "best_mAP"],
            ascending=False,
            na_position="last",
        )
        best_rows.append(sorted_group.iloc[0])
    return pd.DataFrame(best_rows)


def plot_training_curves(train_epoch: pd.DataFrame, best_runs: pd.DataFrame) -> None:
    if train_epoch.empty or best_runs.empty:
        return
    fig = go.Figure()
    for row in best_runs.itertuples():
        subset = train_epoch[
            (train_epoch["experiment"] == row.experiment)
            & (train_epoch["run_id"] == row.run_id)
        ].sort_values("epoch")
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["epoch"],
                y=subset["mean_loss"],
                mode="lines+markers",
                name=row.experiment,
            )
        )
    fig.update_layout(
        title="Training curves (best run per experiment)",
        xaxis_title="Epoch",
        yaxis_title="Mean training loss",
        template="plotly_white",
    )
    fig.write_html(TRAIN_CURVE_HTML, include_plotlyjs="cdn")


def plot_val_metrics(val_df: pd.DataFrame, best_runs: pd.DataFrame) -> None:
    if val_df.empty or best_runs.empty:
        return
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=True,
        subplot_titles=("Validation car mAP", "Validation NDS"),
    )
    for row in best_runs.itertuples():
        subset = val_df[
            (val_df["experiment"] == row.experiment)
            & (val_df["run_id"] == row.run_id)
        ].sort_values("epoch")
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["epoch"],
                y=subset["car_mAP"],
                mode="lines+markers",
                name=row.experiment,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=subset["epoch"],
                y=subset["NDS"],
                mode="lines+markers",
                name=row.experiment,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="car mAP", row=1, col=1)
    fig.update_yaxes(title_text="NDS", row=1, col=2)
    fig.update_layout(template="plotly_white", legend_title_text="Experiment")
    fig.write_html(VAL_CURVE_HTML, include_plotlyjs="cdn")


def plot_best_bar(best_runs: pd.DataFrame) -> None:
    if best_runs.empty:
        return
    metrics = best_runs.sort_values(by=["best_NDS", "best_mAP"], ascending=False)
    fig = go.Figure()
    fig.add_bar(name="car mAP", x=metrics["experiment"], y=metrics["best_car_mAP"])
    fig.add_bar(name="NDS", x=metrics["experiment"], y=metrics["best_NDS"])
    fig.update_layout(
        barmode="group",
        title="Best car mAP/NDS per experiment",
        xaxis_title="Experiment",
        yaxis_title="Score",
        template="plotly_white",
    )
    fig.write_html(BEST_BAR_HTML, include_plotlyjs="cdn")


def collect() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    parsed_logs: List[ParsedLog] = []
    for experiment, run_id, log_path in iter_log_files():
        parsed = parse_log(experiment, run_id, log_path)
        if not parsed.train_rows and not parsed.val_rows:
            continue
        parsed_logs.append(parsed)

    if not parsed_logs:
        print("No usable .log files found.")
        return

    train_rows = [row for log in parsed_logs for row in log.train_rows]
    val_rows = [row for log in parsed_logs for row in log.val_rows]
    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)

    train_epoch = pd.DataFrame()
    if not train_df.empty:
        train_epoch = (
            train_df.groupby(
                ["experiment", "run_id", "run_suffix", "log_path", "epoch"],
                as_index=False,
            )
            .agg(mean_loss=("loss", "mean"))
        )

    best_rows = []
    if not val_df.empty:
        for (exp, run_id), group in val_df.groupby(["experiment", "run_id"]):
            best = choose_best_row(group)
            if best is None:
                continue
            best_rows.append(
                {
                    "experiment": exp,
                    "run_id": run_id,
                    "run_suffix": best["run_suffix"],
                    "best_epoch": int(best["epoch"]),
                    "best_car_mAP": float(best["car_mAP"]) if pd.notna(best["car_mAP"]) else None,
                    "best_mAP": float(best["mAP"]) if pd.notna(best["mAP"]) else None,
                    "best_NDS": float(best["NDS"]) if pd.notna(best["NDS"]) else None,
                    "log_path": best["log_path"],
                }
            )
    best_per_run = pd.DataFrame(
        best_rows,
        columns=[
            "experiment",
            "run_id",
            "run_suffix",
            "best_epoch",
            "best_car_mAP",
            "best_mAP",
            "best_NDS",
            "log_path",
        ],
    )
    if not best_per_run.empty:
        best_per_run.to_csv(CSV_OUTPUT, index=False)
        print(f"Wrote per-run metrics to {CSV_OUTPUT}")
    else:
        print("No validation metrics found to write.")

    if not best_per_run.empty:
        best_runs = pick_best_runs(best_per_run)
    else:
        best_runs = pd.DataFrame(
            columns=[
                "experiment",
                "run_id",
                "run_suffix",
                "best_epoch",
                "best_mAP",
                "best_NDS",
                "log_path",
            ]
        )
    if not best_runs.empty:
        best_runs.to_csv(BEST_BY_EXPERIMENT, index=False)
        print(f"Wrote best-per-experiment metrics to {BEST_BY_EXPERIMENT}")

    plot_training_curves(train_epoch, best_runs)
    plot_val_metrics(val_df, best_runs)
    plot_best_bar(best_runs)
    print(f"Figures saved to {ANALYSIS_DIR}")


if __name__ == "__main__":
    collect()
