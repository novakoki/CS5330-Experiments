"""
Parse MMDetection3D experiment logs under outputs/mmdet, aggregate metrics, and
draw comparison plots for different approaches.

Outputs:
- outputs/mmdet/experiment_metrics.csv: best metrics for every run that has evals
- outputs/mmdet/analysis/best_by_experiment.csv: best run per experiment
- outputs/mmdet/analysis/training_loss.html: mean training loss per epoch
- outputs/mmdet/analysis/val_metrics.html: validation car mAP per epoch
- outputs/mmdet/analysis/best_metric_comparison.html: bar chart of best car mAP
- PNG counterparts for quick embedding: training_loss.png, val_metrics.png, best_metric_comparison.png
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "outputs" / "mmdet"
ANALYSIS_DIR = OUTPUT_ROOT / "analysis"
CSV_OUTPUT = OUTPUT_ROOT / "experiment_metrics.csv"
BEST_BY_EXPERIMENT = ANALYSIS_DIR / "best_by_experiment.csv"
TRAIN_CURVE_HTML = ANALYSIS_DIR / "training_loss.html"
VAL_CURVE_HTML = ANALYSIS_DIR / "val_metrics.html"
BEST_BAR_HTML = ANALYSIS_DIR / "best_metric_comparison.html"
TRAIN_CURVE_PNG = ANALYSIS_DIR / "training_loss.png"
VAL_CURVE_PNG = ANALYSIS_DIR / "val_metrics.png"
BEST_BAR_PNG = ANALYSIS_DIR / "best_metric_comparison.png"

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
]


def save_png_line(
    series: List[Tuple[str, List[List[float]]]],
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
) -> None:
    width, height = 900, 600
    margin = 80
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    points = [p for _, pts in series for p in pts]
    if not points:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_max == y_min:
        y_max += 1
        y_min -= 1

    def to_px(x, y):
        x_norm = (x - x_min) / (x_max - x_min + 1e-9)
        y_norm = (y - y_min) / (y_max - y_min + 1e-9)
        px = margin + x_norm * (width - 2 * margin)
        py = height - margin - y_norm * (height - 2 * margin)
        return px, py

    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    draw.text((margin, margin - 20), title, fill="black", font=font)
    draw.text((width / 2, height - margin + 10), x_label, fill="black", font=font)
    draw.text((10, height / 2), y_label, fill="black", font=font)

    for idx, (label, pts) in enumerate(series):
        if not pts:
            continue
        color = COLORS[idx % len(COLORS)]
        px_pts = [to_px(x, y) for x, y in pts]
        if len(px_pts) > 1:
            draw.line(px_pts, fill=color, width=2)
        for px, py in px_pts:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color, outline=color)
        draw.rectangle(
            (width - margin + 10, margin + idx * 20, width - margin + 25, margin + idx * 20 + 10),
            fill=color,
        )
        draw.text((width - margin + 30, margin + idx * 20 - 3), label, fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_png_bar(
    bars: List[Tuple[str, float]],
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
) -> None:
    width, height = 900, 600
    margin = 100
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    if not bars:
        return
    labels, values = zip(*bars)
    y_min, y_max = 0, max(values) * 1.1 if max(values) > 0 else 1

    def to_px(idx, val, n):
        x = margin + (idx + 0.5) * (width - 2 * margin) / max(n, 1)
        y = height - margin - (val - y_min) / (y_max - y_min + 1e-9) * (height - 2 * margin)
        return x, y

    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    draw.text((margin, margin - 20), title, fill="black", font=font)
    draw.text((width / 2, height - margin + 10), x_label, fill="black", font=font)
    draw.text((10, height / 2), y_label, fill="black", font=font)

    n = len(bars)
    bar_w = (width - 2 * margin) / max(n, 1) * 0.6
    for idx, (label, val) in enumerate(bars):
        x_center, y_top = to_px(idx, val, n)
        x0 = x_center - bar_w / 2
        x1 = x_center + bar_w / 2
        draw.rectangle((x0, y_top, x1, height - margin), fill=COLORS[idx % len(COLORS)])
        draw.text((x_center - 10, height - margin + 5), str(label), fill="black", font=font)
        draw.text((x_center - 10, y_top - 15), f"{val:.3f}", fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

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
    save_png_line(
        series=[
            (
                row.experiment,
                train_epoch[
                    (train_epoch["experiment"] == row.experiment)
                    & (train_epoch["run_id"] == row.run_id)
                ]
                .sort_values("epoch")[["epoch", "mean_loss"]]
                .values.tolist(),
            )
            for row in best_runs.itertuples()
        ],
        title="Training loss (best run per experiment)",
        x_label="Epoch",
        y_label="Mean loss",
        out_path=TRAIN_CURVE_PNG,
    )


def plot_val_metrics(val_df: pd.DataFrame, best_runs: pd.DataFrame) -> None:
    if val_df.empty or best_runs.empty:
        return
    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Validation car mAP",),
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
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="car mAP", row=1, col=1)
    fig.update_layout(template="plotly_white", legend_title_text="Experiment")
    fig.write_html(VAL_CURVE_HTML, include_plotlyjs="cdn")
    save_png_line(
        series=[
            (
                row.experiment,
                val_df[
                    (val_df["experiment"] == row.experiment)
                    & (val_df["run_id"] == row.run_id)
                ]
                .sort_values("epoch")[["epoch", "car_mAP"]]
                .dropna()
                .values.tolist(),
            )
            for row in best_runs.itertuples()
        ],
        title="Validation car mAP (best run per experiment)",
        x_label="Epoch",
        y_label="car mAP",
        out_path=VAL_CURVE_PNG,
    )


def plot_best_bar(best_runs: pd.DataFrame) -> None:
    if best_runs.empty:
        return
    metrics = best_runs.sort_values(by=["best_car_mAP"], ascending=False)
    fig = go.Figure()
    fig.add_bar(name="car mAP", x=metrics["experiment"], y=metrics["best_car_mAP"])
    fig.update_layout(
        barmode="group",
        title="Best car mAP per experiment",
        xaxis_title="Experiment",
        yaxis_title="car mAP",
        template="plotly_white",
    )
    fig.write_html(BEST_BAR_HTML, include_plotlyjs="cdn")
    save_png_bar(
        bars=list(zip(metrics["experiment"], metrics["best_car_mAP"])),
        title="Best car mAP per experiment",
        x_label="Experiment",
        y_label="car mAP",
        out_path=BEST_BAR_PNG,
    )


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

    def keep_latest_for_baseline(df: pd.DataFrame) -> pd.DataFrame:
        """Retain only the latest exp1_baseline run; keep all others."""
        if df.empty:
            return df
        mask = df["experiment"] == "exp1_baseline"
        subset = df[mask]
        if subset.empty:
            return df
        subset = subset.copy()
        subset["_mtime"] = subset["log_path"].map(
            lambda p: Path(p).stat().st_mtime if Path(p).exists() else 0
        )
        latest_idx = subset["_mtime"].idxmax()
        keep_indices = set([latest_idx])
        return df[df.index.isin(keep_indices) | (~mask)]

    if not best_per_run.empty:
        best_per_run = keep_latest_for_baseline(best_per_run)
        best_runs = pick_best_runs(best_per_run)
    else:
        best_runs = pd.DataFrame(
            columns=[
                "experiment",
                "run_id",
                "run_suffix",
                "best_epoch",
                "best_car_mAP",
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
