"""Plot accuracy curves parsed from FoggyTrust-style logs.

Each log is expected to contain lines like:
    [foggytrust - label_flipping_attack] Iteration 109. Test_acc 0.1539
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt

LOG_PATTERN = re.compile(
    r"^\[(?P<method>.+?)\s*-\s*(?P<attack>.+?)\]\s+Iteration\s+"
    r"(?P<iteration>\d+)\.\s+Test_acc\s+(?P<accuracy>\d*\.?\d+)(?:\s+.*)?$"
)


def parse_accuracy_log(log_path: str | Path) -> Tuple[str, List[int], List[float]]:
    """Parse one accuracy log into a label and ordered series."""
    path = Path(log_path)
    if not path.exists() and path.suffix == "":
        txt_path = path.with_suffix(".txt")
        if txt_path.exists():
            path = txt_path
    iterations: List[int] = []
    accuracies: List[float] = []
    label: str | None = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = LOG_PATTERN.match(line)
            if match is None:
                continue

            if label is None:
                method = match.group("method").strip()
                attack = match.group("attack").strip()
                label = f"{method} | {attack}"

            iterations.append(int(match.group("iteration")))
            accuracies.append(float(match.group("accuracy")))

    if not iterations:
        raise ValueError(f"No accuracy records parsed from log: {path}")

    if label is None:
        label = path.stem

    return label, iterations, accuracies


def plot_accuracy_logs(
    log_paths: Iterable[str | Path],
    title: str = "Test Accuracy Over Iterations",
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Aggregate and plot one line per log file."""
    plt.figure(figsize=(11, 6))

    for log_path in log_paths:
        label, iterations, accuracies = parse_accuracy_log(log_path)
        plt.plot(iterations, accuracies, marker="o", markersize=3, linewidth=1.5, label=label)

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Test Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot test accuracy curves from log files.")
    parser.add_argument("log_files", nargs="+", help="Paths to one or more log .txt files.")
    parser.add_argument("--title", default="Test Accuracy Over Iterations", help="Plot title.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output image path (e.g., plotting/accuracy.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display an interactive plot window.",
    )
    args = parser.parse_args()

    plot_accuracy_logs(
        log_paths=args.log_files,
        title=args.title,
        output_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()