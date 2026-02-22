"""
Plot training results from log files.

Usage:
    python plot_results.py path/to/SAM_sgd_resnet18_cifar10.log path/to/ESAM_sgd_resnet18_cifar10.log
    python plot_results.py src/save/*/log/*_wideresnet28_cifar10.log --output results.png
    python plot_results.py src/save/SAM/log/*.log --title "SAM experiments"

Log filenames are expected to follow the pattern:
    [Adaptive_]{sam_type}_{optimizer}_{arch_type}_{dataset}.log
The approach label is derived as "[Adaptive ]{sam_type} ({optimizer})", or "SGD (no SAM)"
for sam_type "none".
"""

import argparse
import re
import os
import sys

import itertools

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
    r"^(?P<adaptive>Adaptive_)?(?P<sam_type>[^_]+)_(?P<optimizer>[^_]+)_(?P<arch_type>.+)_(?P<dataset>[^_]+)$"
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
    adaptive = m.group("adaptive")  # "Adaptive_" or None
    sam_type = m.group("sam_type")
    optimizer = m.group("optimizer")
    if sam_type.lower() in ("none", "standard"):
        return f"{optimizer.upper()} (no SAM)"
    prefix = "Adaptive " if adaptive else ""
    return f"{prefix}{sam_type} ({optimizer})"


def print_summary_table(data: dict, latex: bool = False) -> None:
    rows = [
        (label, d["te_acc"][-1], d["te_loss"][-1], d["epochs"][-1])
        for label, d in data.items()
    ]
    best_acc  = max(r[1] for r in rows)
    best_loss = min(r[2] for r in rows)
    col_w = max(len(r[0]) for r in rows)

    BOLD  = "\033[1m"
    RESET = "\033[0m"

    header = f"{'Method':<{col_w}}  {'Val Acc (%)':>12}  {'Val Loss':>10}  {'Epoch':>6}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for label, acc, loss, epoch in rows:
        acc_str  = f"{acc:12.2f}"
        loss_str = f"{loss:10.4f}"
        if acc  == best_acc:
            acc_str  = BOLD + acc_str  + RESET
        if loss == best_loss:
            loss_str = BOLD + loss_str + RESET
        print(f"{label:<{col_w}}  {acc_str}  {loss_str}  {epoch:>6d}")
    print(sep + "\n")

    if latex:
        print("% LaTeX table (requires \\usepackage{booktabs})")
        print(r"\begin{tabular}{lrrr}")
        print(r"\toprule")
        print(r"Method & Val Acc (\%) & Val Loss & Epoch \\")
        print(r"\midrule")
        for label, acc, loss, epoch in rows:
            acc_str  = f"\\textbf{{{acc:.2f}}}" if acc  == best_acc  else f"{acc:.2f}"
            loss_str = f"\\textbf{{{loss:.4f}}}" if loss == best_loss else f"{loss:.4f}"
            print(f"{label} & {acc_str} & {loss_str} & {epoch} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}" + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot SAM training results.")
    parser.add_argument("log_files", nargs="+",
                        help="One or more log file paths (shell globs work)")
    parser.add_argument("--title", default=None,
                        help="Custom figure title (default: auto-detected from filenames)")
    parser.add_argument("--output", default=None,
                        help="Save figure to this path instead of showing interactively")
    parser.add_argument("--table", action="store_true",
                        help="Print a summary table of last-epoch validation metrics")
    parser.add_argument("--latex", action="store_true",
                        help="Also print the summary table as LaTeX (implies --table)")
    args = parser.parse_args()
    if args.latex:
        args.table = True

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

    # Sort labels into preferred display order; unknown labels go at the end.
    LABEL_ORDER = [
        "SGD (no SAM)",
        "ADAM (no SAM)",
        "ADAMW (no SAM)",
        "SAM (sgd)",
        "SAM (adam)",
        "SAM (adamw)",
        "Adaptive SAM (sgd)",
        "Adaptive SAM (adam)",
        "Adaptive SAM (adamw)",
    ]
    def _sort_key(label):
        try:
            return LABEL_ORDER.index(label)
        except ValueError:
            return len(LABEL_ORDER)
    data = dict(sorted(data.items(), key=lambda kv: _sort_key(kv[0])))

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

    linestyles = ["-", "--", "-.", ":"]
    style_cycle = itertools.cycle(linestyles)
    label_styles = {label: next(style_cycle) for label in data}

    for label, d in data.items():
        ls = label_styles[label]
        for key, ax in axes.items():
            ax.plot(d["epochs"], d[key], label=label, linewidth=1.5, linestyle=ls)

    for key, ax in axes.items():
        ax.set_title(titles[key], fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    if args.table:
        print_summary_table(data, latex=args.latex)

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
