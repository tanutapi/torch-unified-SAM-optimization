"""
Plot training results from log files.

Usage:
    python plot_results.py path/to/SAM_sgd_resnet18_cifar10.log path/to/ESAM_sgd_resnet18_cifar10.log
    python plot_results.py src/save/*/log/*_wideresnet28_cifar10.log --output results.png
    python plot_results.py src/save/SAM/log/*.log --title "SAM experiments"

Log filenames are expected to follow the pattern:
    {sam_type}_{optimizer}_{arch_type}_{dataset}.log
The approach label is derived as "{sam_type} ({optimizer})", or "SGD (no SAM)" for
sam_type "none".
"""

import argparse
import re
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


LOG_PATTERN = re.compile(
    r"Epoch:\s*(\d+)/\d+\s*\|"
    r"\s*Train Loss:\s*([\d.]+)\s*\|"
    r"\s*Train Acc:\s*([\d.]+)%\s*\|"
    r"\s*Test Loss:\s*([\d.]+)\s*\|"
    r"\s*Test Acc:\s*([\d.]+)%"
)

FILENAME_PATTERN = re.compile(
    r"^(?P<sam_type>.+?)_(?P<optimizer>sgd|adam|adamw)_(?P<arch_type>.+)_(?P<dataset>[^_]+)$",
    re.IGNORECASE,
)


def parse_log(path: str) -> dict:
    epochs, tr_loss, tr_acc, te_loss, te_acc = [], [], [], [], []
    with open(path) as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if m:
                epochs.append(int(m.group(1)))
                tr_loss.append(float(m.group(2)))
                tr_acc.append(float(m.group(3)))
                te_loss.append(float(m.group(4)))
                te_acc.append(float(m.group(5)))
    return {"epochs": epochs, "tr_loss": tr_loss, "tr_acc": tr_acc,
            "te_loss": te_loss, "te_acc": te_acc}


def label_from_filename(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    m = FILENAME_PATTERN.match(stem)
    if not m:
        return stem
    sam_type = m.group("sam_type")
    optimizer = m.group("optimizer")
    if sam_type.lower() in ("none", "standard"):
        return f"{optimizer.upper()} (no SAM)"
    return f"{sam_type} ({optimizer})"


def main():
    parser = argparse.ArgumentParser(description="Plot SAM training results.")
    parser.add_argument("log_files", nargs="+",
                        help="One or more log file paths (shell globs work)")
    parser.add_argument("--title", default=None,
                        help="Custom figure title (default: auto-detected from filenames)")
    parser.add_argument("--output", default=None,
                        help="Save figure to this path instead of showing interactively")
    args = parser.parse_args()

    # Load data
    data = {}
    for log_path in args.log_files:
        if not os.path.exists(log_path):
            print(f"[warning] log not found: {log_path}", file=sys.stderr)
            continue
        parsed = parse_log(log_path)
        if not parsed["epochs"]:
            print(f"[warning] no data parsed from: {log_path}", file=sys.stderr)
            continue
        label = label_from_filename(log_path)
        data[label] = parsed

    if not data:
        print("No data loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Auto-detect title from filenames if not provided
    if args.title:
        title = args.title
    else:
        stems = [os.path.splitext(os.path.basename(p))[0] for p in args.log_files]
        parts = [FILENAME_PATTERN.match(s) for s in stems]
        arch_types = {m.group("arch_type") for m in parts if m}
        datasets = {m.group("dataset") for m in parts if m}
        arch_str = ", ".join(sorted(arch_types)) if arch_types else "unknown"
        data_str = ", ".join(sorted(datasets)) if datasets else "unknown"
        title = f"Training Results — {arch_str} on {data_str}"

    # Plot
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    axes = {
        "tr_loss": fig.add_subplot(gs[0, 0]),
        "tr_acc":  fig.add_subplot(gs[0, 1]),
        "te_loss": fig.add_subplot(gs[1, 0]),
        "te_acc":  fig.add_subplot(gs[1, 1]),
    }

    titles = {
        "tr_loss": "Training Loss",
        "tr_acc":  "Training Accuracy (%)",
        "te_loss": "Validation Loss",
        "te_acc":  "Validation Accuracy (%)",
    }

    for label, d in data.items():
        for key, ax in axes.items():
            ax.plot(d["epochs"], d[key], label=label, linewidth=1.5)

    for key, ax in axes.items():
        ax.set_title(titles[key], fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
