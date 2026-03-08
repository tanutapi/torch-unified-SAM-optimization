"""
Plot training results from log files.

Usage:
    python plot_results.py path/to/SAM_sgd_resnet18_cifar10.log path/to/ESAM_sgd_resnet18_cifar10.log
    python plot_results.py src/save/*/log/*_wideresnet28_cifar10.log --output results.png
    python plot_results.py src/save/SAM/log/*.log --title "SAM experiments"

Log filenames are expected to follow the pattern:
    [Adaptive_]{sam_type}_{optimizer}_{arch_type}_{dataset}.log
The approach label is derived as "[ASAM|SAM] {opt_name}[ 8Bits]", or just "{opt_name}[ 8Bits]"
for sam_type "none".
"""

import argparse
import re
import os
import sys
import itertools
from collections import defaultdict

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

# Display name for each base optimizer
OPT_DISPLAY = {"sgd": "SGD", "adam": "Adam", "adamw": "AdamW"}

# Full name and short name for table section headers
OPTIMIZER_DISPLAY = {
    "sgd":   ("Stochastic Gradient Descent Optimizer", "SGD"),
    "adam":  ("Adaptive Moment Estimation Optimizer", "Adam"),
    "adamw": ("Adam with Weight Decay", "AdamW"),
}

# Column definitions: (sam_key, is_8bit, label_template)
# sam_key: "none" | "sam" | "asam"
COLUMN_DEFS = [
    ("none", False, "{opt}"),
    ("sam",  False, "{opt} SAM"),
    ("asam", False, "{opt} ASAM"),
    ("none", True,  "{opt} 8Bits"),
    ("sam",  True,  "{opt} SAM 8Bits"),
    ("asam", True,  "{opt} ASAM 8Bits"),
]


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
    adaptive = m.group("adaptive")
    sam_type = m.group("sam_type")
    optimizer = m.group("optimizer").lower()
    is_8bit = optimizer.endswith("8bit")
    base_opt = optimizer[:-4] if is_8bit else optimizer
    opt_name = OPT_DISPLAY.get(base_opt, base_opt.upper())
    suffix = " 8Bits" if is_8bit else ""
    if sam_type.lower() in ("none", "standard"):
        return f"{opt_name}{suffix}"
    sam_prefix = "ASAM" if adaptive else "SAM"
    return f"{opt_name} {sam_prefix}{suffix}"


def meta_from_filename(path: str) -> dict:
    """Extract structured metadata (base optimizer, SAM variant, 8-bit flag) from log filename."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = FILENAME_PATTERN.match(stem)
    if not m:
        return {}
    adaptive = bool(m.group("adaptive"))
    sam_type = m.group("sam_type").lower()
    optimizer = m.group("optimizer").lower()
    is_8bit = optimizer.endswith("8bit")
    base_opt = optimizer[:-4] if is_8bit else optimizer
    if sam_type in ("none", "standard"):
        sam_key = "none"
    elif adaptive:
        sam_key = "asam"
    else:
        sam_key = "sam"
    return {"base_opt": base_opt, "is_8bit": is_8bit, "sam_key": sam_key}


def print_summary_table(data: dict, meta_dict: dict, latex: bool = False) -> None:
    BOLD  = "\033[1m"
    RESET = "\033[0m"

    # Group entries by base optimizer
    # groups[base_opt][(sam_key, is_8bit)] = (te_acc, te_loss, epoch_count)
    groups = defaultdict(dict)
    ungrouped = []

    for label, d in data.items():
        meta = meta_dict.get(label, {})
        if not d["te_acc"]:
            continue
        entry = (d["te_acc"][-1], d["te_loss"][-1], d["epochs"][-1])
        if meta and "base_opt" in meta:
            groups[meta["base_opt"]][(meta["sam_key"], meta["is_8bit"])] = entry
        else:
            ungrouped.append((label, *entry))

    if latex:
        print("\n% LaTeX table (requires \\usepackage{booktabs})")

    latex_sections = []

    # Print one grouped table per base optimizer (console + optional LaTeX together)
    for base_opt in ["sgd", "adam", "adamw"]:
        if base_opt not in groups:
            continue
        group = groups[base_opt]
        long_name, short_name = OPTIMIZER_DISPLAY.get(base_opt, (base_opt.upper(), base_opt.upper()))
        epochs = [v[2] for v in group.values()]
        epoch_str = str(max(epochs)) if epochs else "?"

        col_labels = [spec[2].format(opt=short_name) for spec in COLUMN_DEFS]
        col_keys   = [(spec[0], spec[1]) for spec in COLUMN_DEFS]
        col_data   = [group.get(k) for k in col_keys]

        avail = [v for v in col_data if v is not None]
        best_acc  = max(v[0] for v in avail) if avail else None
        best_loss = min(v[1] for v in avail) if avail else None

        row_hdr_w  = max(len("Validation"), len("Val Acc (%)"))
        col_widths = [max(len(cl), 8) for cl in col_labels]

        # --- Console table (skipped when --latex is active) ---
        if not latex:
            hdr = f"{'Validation':<{row_hdr_w}}"
            for cl, cw in zip(col_labels, col_widths):
                hdr += f" | {cl:<{cw}}"
            sep = "-" * len(hdr)
            print(f"\n{long_name} - {short_name} ({epoch_str} Epoch)")
            print(sep)
            print(hdr)
            print(sep)

            row = f"{'Val Loss':<{row_hdr_w}}"
            for v, cw in zip(col_data, col_widths):
                if v is None:
                    cell = f"{'N/A':<{cw}}"
                else:
                    plain = f"{v[1]:.4f}"
                    cell  = f"{plain:<{cw}}"
                    if v[1] == best_loss:
                        cell = BOLD + cell + RESET
                row += f" | {cell}"
            print(row)

            row = f"{'Val Acc (%)':<{row_hdr_w}}"
            for v, cw in zip(col_data, col_widths):
                if v is None:
                    cell = f"{'N/A':<{cw}}"
                else:
                    plain = f"{v[0]:.2f}"
                    cell  = f"{plain:<{cw}}"
                    if v[0] == best_acc:
                        cell = BOLD + cell + RESET
                row += f" | {cell}"
            print(row)
            print(sep)

        # Collect rendered LaTeX rows for later (printed as one unified table)
        if latex:
            def _fmt_acc(v, _best=best_acc):
                if v is None:
                    return r"\text{--}"
                s = f"{v[0]:.2f}"
                return f"\\textbf{{{s}}}" if v[0] == _best else s

            def _fmt_loss(v, _best=best_loss):
                if v is None:
                    return r"\text{--}"
                s = f"{v[1]:.4f}"
                return f"\\textbf{{{s}}}" if v[1] == _best else s

            latex_sections.append({
                "header":    col_labels,
                "loss_row":  [_fmt_loss(v) for v in col_data],
                "acc_row":   [_fmt_acc(v)  for v in col_data],
                "col_spec":  "l" + "r" * len(col_labels),
                "caption":   f"{long_name} - {short_name} ({epoch_str} Epoch)",
            })

    # Print unified LaTeX table (all optimizer groups in one tabular)
    if latex and latex_sections:
        col_spec = latex_sections[0]["col_spec"]
        print(f"\\begin{{tabular}}{{{col_spec}}}")
        print(r"\toprule")
        for i, sec in enumerate(latex_sections):
            if i > 0:
                print(r"\midrule")
            print(r"\midrule")
            print(f"Validation & {' & '.join(sec['header'])} \\\\")
            print(r"\midrule")
            print("Val Loss & "    + " & ".join(sec["loss_row"]) + r" \\")
            print(r"Val Acc (\%) & " + " & ".join(sec["acc_row"])  + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabular}")

    # Fallback: print any entries that could not be grouped (console only)
    if ungrouped and not latex:
        print("\n(Ungrouped entries):")
        col_w = max(len(r[0]) for r in ungrouped)
        header = f"{'Method':<{col_w}}  {'Val Acc (%)':>12}  {'Val Loss':>10}  {'Epoch':>6}"
        sep = "-" * len(header)
        best_acc  = max(r[1] for r in ungrouped)
        best_loss = min(r[2] for r in ungrouped)
        print(sep); print(header); print(sep)
        for label, acc, loss, epoch in ungrouped:
            acc_str  = f"{acc:12.2f}"
            loss_str = f"{loss:10.4f}"
            if acc  == best_acc:  acc_str  = BOLD + acc_str  + RESET
            if loss == best_loss: loss_str = BOLD + loss_str + RESET
            print(f"{label:<{col_w}}  {acc_str}  {loss_str}  {epoch:>6d}")
        print(sep)


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
    meta_dict = {}
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
        meta_dict[label] = meta_from_filename(log_path)

    if not data:
        print("No data loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Sort labels into preferred display order; unknown labels go at the end.
    LABEL_ORDER = [
        "SGD",        "SGD SAM",        "SGD ASAM",
        "SGD 8Bits",  "SGD SAM 8Bits",  "SGD ASAM 8Bits",
        "Adam",       "Adam SAM",       "Adam ASAM",
        "Adam 8Bits", "Adam SAM 8Bits", "Adam ASAM 8Bits",
        "AdamW",      "AdamW SAM",      "AdamW ASAM",
        "AdamW 8Bits","AdamW SAM 8Bits","AdamW ASAM 8Bits",
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
        print_summary_table(data, meta_dict, latex=args.latex)

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
