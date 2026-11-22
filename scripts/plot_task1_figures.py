#!/usr/bin/env python3
"""scripts/plot_task1_figures.py — Generate publication-ready figures for Task 1 report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Styling configuration for NeurIPS-style figures
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

RESULTS_DIR = Path("runs/task1_results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    with open(RESULTS_DIR / "all_results.json", "r", encoding="utf-8") as f:
        res = json.load(f)
    with open(RESULTS_DIR / "bootstrap_ci.json", "r", encoding="utf-8") as f:
        ci = json.load(f)
    return res, ci


def plot_sparsity_tradeoff(res):
    """Figure 1: Quality vs. Global Sparsity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    base_ppl = res["baseline_fp16"]["wikitext_ppl"]
    base_hs = res["baseline_fp16"]["hellaswag_acc_norm"] * 100

    # Pruning data
    sparsities = [0.0, 0.25, 0.50, 0.75]

    # P1 Magnitude
    p1_ppl = [base_ppl, res["P1_magnitude_s25"]["wikitext_ppl"], res["P1_magnitude_s50"]["wikitext_ppl"], res["P1_magnitude_s75"]["wikitext_ppl"]]
    p1_hs = [base_hs, res["P1_magnitude_s25"]["hellaswag_acc_norm"] * 100, res["P1_magnitude_s50"]["hellaswag_acc_norm"] * 100, res["P1_magnitude_s75"]["hellaswag_acc_norm"] * 100]

    # P2 Wanda
    p2_ppl = [base_ppl, res["P2_wanda_s25"]["wikitext_ppl"], res["P2_wanda_s50"]["wikitext_ppl"], res["P2_wanda_s75"]["wikitext_ppl"]]
    p2_hs = [base_hs, res["P2_wanda_s25"]["hellaswag_acc_norm"] * 100, res["P2_wanda_s50"]["hellaswag_acc_norm"] * 100, res["P2_wanda_s75"]["hellaswag_acc_norm"] * 100]

    # P3 Non-Uniform
    p3_sparsities = [0.0, 0.50, 0.75]
    p3_ppl = [base_ppl, res["P3_nonuniform_s50"]["wikitext_ppl"], res["P3_nonuniform_s75"]["wikitext_ppl"]]
    p3_hs = [base_hs, res["P3_nonuniform_s50"]["hellaswag_acc_norm"] * 100, res["P3_nonuniform_s75"]["hellaswag_acc_norm"] * 100]

    # 2:4 Structured
    p_24_ppl = res["P_structured_2_4"]["wikitext_ppl"]
    p_24_hs = res["P_structured_2_4"]["hellaswag_acc_norm"] * 100

    # Subplot 1: WikiText Perplexity (log scale)
    ax1.plot(sparsities, p1_ppl, marker="o", color="#d62728", label="P1: Magnitude", linestyle="-")
    ax1.plot(sparsities, p2_ppl, marker="s", color="#1f77b4", label="P2: Wanda", linestyle="-")
    ax1.plot(p3_sparsities, p3_ppl, marker="^", color="#2ca02c", label="P3: Non-Uniform Wanda", linestyle="--")
    ax1.scatter([0.50], [p_24_ppl], marker="*", color="#9467bd", s=180, zorder=5, label="Structured 2:4 (50%)")

    ax1.set_yscale("log")
    ax1.set_xlabel("Global Sparsity")
    ax1.set_ylabel("WikiText-2 Perplexity (Log Scale)")
    ax1.set_title("Perplexity vs. Sparsity (Lower is Better)")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    # Subplot 2: HellaSwag acc_norm
    ax2.plot(sparsities, p1_hs, marker="o", color="#d62728", label="P1: Magnitude", linestyle="-")
    ax2.plot(sparsities, p2_hs, marker="s", color="#1f77b4", label="P2: Wanda", linestyle="-")
    ax2.plot(p3_sparsities, p3_hs, marker="^", color="#2ca02c", label="P3: Non-Uniform Wanda", linestyle="--")
    ax2.scatter([0.50], [p_24_hs], marker="*", color="#9467bd", s=180, zorder=5, label="Structured 2:4 (50%)")

    ax2.set_xlabel("Global Sparsity")
    ax2.set_ylabel("Zero-Shot HellaSwag acc_norm (%)")
    ax2.set_title("Downstream Accuracy vs. Sparsity (Higher is Better)")
    ax2.grid(True)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "task1_sparsity_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "task1_sparsity_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved task1_sparsity_tradeoff.pdf & .png")


def plot_quantization_tradeoff(res):
    """Figure 2: Quality vs. Packed Model Storage (MB)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    base_mb = res["baseline_fp16"]["packed_mb"]
    base_ppl = res["baseline_fp16"]["wikitext_ppl"]
    base_hs = res["baseline_fp16"]["hellaswag_acc_norm"] * 100

    # Q1 Uniform
    q1_mb = [res["Q1_uniform_int8"]["packed_mb"], res["Q1_uniform_int4"]["packed_mb"], res["Q1_uniform_int3"]["packed_mb"]]
    q1_ppl = [res["Q1_uniform_int8"]["wikitext_ppl"], res["Q1_uniform_int4"]["wikitext_ppl"], res["Q1_uniform_int3"]["wikitext_ppl"]]
    q1_hs = [res["Q1_uniform_int8"]["hellaswag_acc_norm"] * 100, res["Q1_uniform_int4"]["hellaswag_acc_norm"] * 100, res["Q1_uniform_int3"]["hellaswag_acc_norm"] * 100]

    # Q2 AWQ
    q2_mb = [res["Q2_awq_int8"]["packed_mb"], res["Q2_awq_int4"]["packed_mb"], res["Q2_awq_int3"]["packed_mb"]]
    q2_ppl = [res["Q2_awq_int8"]["wikitext_ppl"], res["Q2_awq_int4"]["wikitext_ppl"], res["Q2_awq_int3"]["wikitext_ppl"]]
    q2_hs = [res["Q2_awq_int8"]["hellaswag_acc_norm"] * 100, res["Q2_awq_int4"]["hellaswag_acc_norm"] * 100, res["Q2_awq_int3"]["hellaswag_acc_norm"] * 100]

    # Q3 Mixed Precision
    q3_mb = [res["Q3_mixed_match_int4"]["packed_mb"], res["Q3_mixed_match_int3"]["packed_mb"]]
    q3_ppl = [res["Q3_mixed_match_int4"]["wikitext_ppl"], res["Q3_mixed_match_int3"]["wikitext_ppl"]]
    q3_hs = [res["Q3_mixed_match_int4"]["hellaswag_acc_norm"] * 100, res["Q3_mixed_match_int3"]["hellaswag_acc_norm"] * 100]

    # Subplot 1: WikiText Perplexity
    ax1.plot([base_mb] + q1_mb, [base_ppl] + q1_ppl, marker="o", color="#e377c2", label="Q1: Uniform PTQ", linestyle="-")
    ax1.plot([base_mb] + q2_mb, [base_ppl] + q2_ppl, marker="s", color="#17becf", label="Q2: AWQ", linestyle="-")
    ax1.scatter(q3_mb, q3_ppl, marker="D", color="#ff7f0e", s=100, zorder=5, label="Q3: Mixed Precision")

    ax1.set_xlabel("Packed Storage (MB)")
    ax1.set_ylabel("WikiText-2 Perplexity")
    ax1.set_title("Perplexity vs. Model Storage (Lower is Better)")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    # Subplot 2: HellaSwag acc_norm
    ax2.plot([base_mb] + q1_mb, [base_hs] + q1_hs, marker="o", color="#e377c2", label="Q1: Uniform PTQ", linestyle="-")
    ax2.plot([base_mb] + q2_mb, [base_hs] + q2_hs, marker="s", color="#17becf", label="Q2: AWQ", linestyle="-")
    ax2.scatter(q3_mb, q3_hs, marker="D", color="#ff7f0e", s=100, zorder=5, label="Q3: Mixed Precision")

    ax2.set_xlabel("Packed Storage (MB)")
    ax2.set_ylabel("Zero-Shot HellaSwag acc_norm (%)")
    ax2.set_title("Downstream Accuracy vs. Model Storage (Higher is Better)")
    ax2.grid(True)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "task1_quantization_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "task1_quantization_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved task1_quantization_tradeoff.pdf & .png")


def plot_bootstrap_ci(ci):
    """Figure 3: Forest plot of 95% Paired Bootstrap Confidence Intervals."""
    comparisons = [
        ("Q3 vs Q2 (INT4 Budget)", "Q3_vs_Q2_int4_hellaswag_acc_norm", "% acc_norm"),
        ("Q3 vs Q2 (INT3 Budget)", "Q3_vs_Q2_int3_hellaswag_acc_norm", "% acc_norm"),
        ("P3 vs P2 (50% Sparsity)", "P3_vs_P2_50_hellaswag_acc_norm", "% acc_norm"),
        ("P3 vs P2 (75% Sparsity)", "P3_vs_P2_75_hellaswag_acc_norm", "% acc_norm"),
    ]

    labels = []
    means = []
    ci_lows = []
    ci_highs = []

    for label, key, _ in comparisons:
        item = ci.get(key)
        if item is not None:
            labels.append(label)
            # Convert to percentage points
            means.append(item["mean_diff"] * 100)
            ci_lows.append(item["ci_low"] * 100)
            ci_highs.append(item["ci_high"] * 100)

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(labels))

    err_low = np.array(means) - np.array(ci_lows)
    err_high = np.array(ci_highs) - np.array(means)

    ax.errorbar(means, y_pos, xerr=[err_low, err_high], fmt="o", color="#1f77b4",
                ecolor="#1f77b4", elinewidth=2.5, capsize=6, capthick=2, markersize=8)

    ax.axvline(0, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired Difference in Zero-Shot HellaSwag acc_norm (Percentage Points, 95% CI)")
    ax.set_title("Statistical Paired Bootstrap Analysis (B = 2,000 Resamples)")
    ax.grid(True, axis="x")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "task1_bootstrap_ci.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "task1_bootstrap_ci.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved task1_bootstrap_ci.pdf & .png")


def main():
    print("=" * 80)
    print("Generating Task 1 Report Figures...")
    print("=" * 80)
    res, ci = load_results()
    plot_sparsity_tradeoff(res)
    plot_quantization_tradeoff(res)
    plot_bootstrap_ci(ci)
    print("\n✓ All Task 1 figures successfully generated into figures/.")


if __name__ == "__main__":
    main()
