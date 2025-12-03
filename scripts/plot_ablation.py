"""Plot bar chart comparing the four experiments."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON mapping exp_name -> mAP")
    parser.add_argument("--output", default="plots/ablation.png")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        results = json.load(f)
    names = list(results.keys())
    maps = [results[k] for k in names]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(names, maps, color=["gray", "orange", "blue", "green"])
    plt.ylabel("mAP@0.5")
    plt.title("Ablation Study")
    plt.ylim(0, max(maps + [0.5]))
    for bar, score in zip(bars, maps):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{score:.2f}", ha="center", va="bottom")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved ablation plot to {args.output}")


if __name__ == "__main__":
    main()

