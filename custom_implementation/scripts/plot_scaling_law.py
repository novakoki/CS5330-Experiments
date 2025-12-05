"""Plot mAP vs dataset size for scaling experiment."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON list of {'size': int, 'map': float}")
    parser.add_argument("--output", default="plots/scaling_law.png")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)
    sizes = [d["size"] for d in data]
    maps = [d["map"] for d in data]

    plt.figure(figsize=(6, 4))
    plt.plot(sizes, maps, marker="o")
    plt.xlabel("Training scenes")
    plt.ylabel("mAP@0.5")
    plt.title("Scaling Law: Data vs Performance")
    plt.grid(True, linestyle="--", alpha=0.4)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()

