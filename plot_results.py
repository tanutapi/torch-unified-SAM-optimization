"""
Plot training results from log files in src/save/{sam_type}/log/.

Usage:
    python plot_results.py SAM,ESAM,FisherSAM --arch_type wideresnet28 --dataset cifar10
    python plot_results.py SAM,none            --arch_type resnet18     --dataset cifar100
    python plot_results.py SAM                 --arch_type resnet18     --dataset cifar10 --output results.png
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


def find_log(sam_type: str, optimizer: str, arch_type: str, dataset: str) -> str:
    sam_dir = sam_type if sam_type.lower() not in ("none", "standard", "") else "standard"
    path = os.path.join("src", "save", sam_dir, "log", f"{sam_dir}_{optimizer}_{arch_type}_{dataset}.log")
    return path


def main():
    parser = argparse.ArgumentParser(description="Plot SAM training results.")
    parser.add_argument("sam_types", type=str,
                        help="Comma-separated SAM variant names, e.g. SAM,ESAM,none")
    parser.add_argument("--optimizer", default="sgd",
                        help="Base optimizer used during training, e.g. sgd, adam, adamw")
    parser.add_argument("--arch_type", required=True,
                        help="Model architecture, e.g. wideresnet28")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name, e.g. cifar10")
    parser.add_argument("--output", default=None,
                        help="Save figure to this path instead of showing interactively")
    args = parser.parse_args()

    variants = [v.strip() for v in args.sam_types.split(",") if v.strip()]

    # Load data
    data = {}
    for v in variants:
        log_path = find_log(v, args.optimizer, args.arch_type, args.dataset)
        if not os.path.exists(log_path):
            print(f"[warning] log not found: {log_path}", file=sys.stderr)
            continue
        parsed = parse_log(log_path)
        if not parsed["epochs"]:
            print(f"[warning] no data parsed from: {log_path}", file=sys.stderr)
            continue
        data[v] = parsed

    if not data:
        print("No data loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Plot
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Training Results — {args.arch_type} on {args.dataset}",
        fontsize=14, fontweight="bold", y=0.98,
    )
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

    for variant, d in data.items():
        label = variant if variant.lower() not in ("none", "standard") else "SGD (no SAM)"
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
