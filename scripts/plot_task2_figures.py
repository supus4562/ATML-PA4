"""scripts/plot_task2_figures.py — Generates publication-ready figures for Task 2.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "runs" / "task2_results"
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Course paper style settings
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "font.family": "serif",
    "figure.autolayout": True,
})


def plot_attention_operators(data: dict):
    """Plot Part A operator latency, numerical error, and VRAM scaling."""
    grid = data.get("grid", {}).get("grid", [])
    if not grid:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    lengths = sorted(list(set(r["length"] for r in grid)))
    ops = ["dense", "tiled", "sdpa"]
    colors = {"dense": "#e74c3c", "tiled": "#2980b9", "sdpa": "#27ae60"}
    markers = {"dense": "o", "tiled": "s", "sdpa": "^"}

    # Subplot 1: Forward Latency (FP16)
    ax1 = axes[0]
    for op in ops:
        sub = [r for r in grid if r["operator"] == op and r["dtype"] == "fp16"]
        if sub:
            xs = [r["length"] for r in sub]
            ys = [r["fwd_latency_ms"] for r in sub]
            ax1.plot(xs, ys, marker=markers[op], color=colors[op], label=op.upper(), linewidth=1.8)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Sequence Length L")
    ax1.set_ylabel("Forward Latency (ms)")
    ax1.set_title("Forward Latency vs. Length (FP16)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Subplot 2: Output Error vs Dense FP32 Reference
    ax2 = axes[1]
    for op in ["tiled", "sdpa"]:
        sub = [r for r in grid if r["operator"] == op and r["dtype"] == "fp16"]
        if sub:
            xs = [r["length"] for r in sub]
            ys = [max(1e-7, r["output_err_vs_fp32"]) for r in sub]
            ax2.plot(xs, ys, marker=markers[op], color=colors[op], label=f"{op.upper()} (FP16)", linewidth=1.8)
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("Sequence Length L")
    ax2.set_ylabel("Max Absolute Error vs. FP32 Ref")
    ax2.set_title("Numerical Error vs. Reference")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # Subplot 3: Peak Allocated VRAM
    ax3 = axes[2]
    for op in ops:
        sub = [r for r in grid if r["operator"] == op and r["dtype"] == "fp16"]
        if sub:
            xs = [r["length"] for r in sub]
            ys = [r["peak_vram_mb"] for r in sub]
            ax3.plot(xs, ys, marker=markers[op], color=colors[op], label=op.upper(), linewidth=1.8)
    ax3.set_xscale("log", base=2)
    ax3.set_xlabel("Sequence Length L")
    ax3.set_ylabel("Peak VRAM (MB)")
    ax3.set_title("Peak Allocated VRAM vs. Length")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    out_path = FIGURES_DIR / "task2_attention_operators.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved {out_path}")


def plot_mqar_results(part_b_data: dict):
    """Plot Part B training loss curves and distance accuracy heatmaps."""
    if not part_b_data:
        return

    # 1. Training Curves
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {
        "dense": "#e74c3c",
        "window": "#3498db",
        "linear": "#f39c12",
        "dilated": "#9b59b6",
        "dgla": "#2ecc71",
    }

    # Aggregate loss curves across seeds for K=16 mixed
    for attn_type in ["dense", "window", "linear", "dilated", "dgla"]:
        curves = []
        for seed in [6304, 6305, 6306]:
            k_key = f"{attn_type}_k16_mixed_seed{seed}"
            if k_key in part_b_data:
                loss_pts = part_b_data[k_key]["train_summary"].get("loss_curve", [])
                if loss_pts:
                    curves.append([pt[1] for pt in loss_pts])
        if curves:
            steps = [pt[0] for pt in part_b_data[f"{attn_type}_k16_mixed_seed6304"]["train_summary"]["loss_curve"]]
            arr = np.array(curves)
            mean_loss = np.mean(arr, axis=0)
            std_loss = np.std(arr, axis=0)
            ax.plot(steps, mean_loss, label=f"{attn_type.upper()} (K=16)", color=colors[attn_type], linewidth=1.8)
            ax.fill_between(steps, mean_loss - std_loss, mean_loss + std_loss, color=colors[attn_type], alpha=0.15)

    ax.set_xlabel("Optimizer Updates")
    ax.set_ylabel("Cross-Entropy Loss (64 Classes)")
    ax.set_title("MQAR Training Loss Curves (K=16, Mixed Regime, Mean ± 1 Std across 3 Seeds)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    out_curves = FIGURES_DIR / "task2_mqar_learning_curves.png"
    plt.savefig(out_curves, dpi=300)
    plt.close()
    print(f"✓ Saved {out_curves}")

    # 2. Distance Bin Heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = ["[0, 32)", "[32, 64)", "[64, 128)", "[128, 256)"]
    models = ["dense", "window", "linear", "dilated", "dgla"]
    matrix = np.zeros((len(models), len(bins)))

    for m_idx, m_name in enumerate(models):
        k_key = f"{m_name}_k16_mixed_seed6304"
        if k_key in part_b_data:
            eval_res = part_b_data[k_key]["eval_metrics"].get("mixed", {})
            dist_bins = eval_res.get("distance_bins", {})
            for b_idx, b_label in enumerate(bins):
                matrix[m_idx, b_idx] = dist_bins.get(b_label, {}).get("accuracy", 0.0) * 100.0

    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.upper() for m in models])
    ax.set_xlabel("Source-Query Distance Bin")
    ax.set_ylabel("Attention Architecture")
    ax.set_title("MQAR Accuracy (%) by Distance Bin (K=16 Mixed)")

    for i in range(len(models)):
        for j in range(len(bins)):
            val = matrix[i, j]
            text_color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    out_heatmap = FIGURES_DIR / "task2_mqar_distance_heatmap.png"
    plt.savefig(out_heatmap, dpi=300)
    plt.close()
    print(f"✓ Saved {out_heatmap}")


def plot_kv_results(part_c_data: dict):
    """Plot Part C KV cache trade-offs and paired bootstrap confidence intervals."""
    summary = part_c_data.get("summary", {})
    bootstrap = part_c_data.get("bootstrap", {})
    if not summary:
        return

    # 1. Quality vs Packed MB
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    policies = list(summary.keys())
    colors = ["#2c3e50", "#e74c3c", "#3498db", "#9b59b6", "#27ae60", "#f39c12"]

    ax1 = axes[0]
    for idx, p in enumerate(policies):
        acc = summary[p]["exact_match_accuracy"] * 100.0
        mb = summary[p]["mean_packed_mb"]
        ax1.scatter(mb, acc, color=colors[idx % len(colors)], s=120, label=p.replace("_", " ").title())
        ax1.annotate(p.split("_")[0].upper(), (mb, acc), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax1.set_xlabel("Packed KV Cache (MB)")
    ax1.set_ylabel("Exact Match Retrieval Accuracy (%)")
    ax1.set_title("Retrieval Quality vs. KV Cache Size")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # 2. Forest plot of Paired 95% Bootstrap CIs
    ax2 = axes[1]
    ci_keys = [k for k in bootstrap.keys() if k.endswith("_acc")]
    labels = [k.replace("_vs_question_aware_acc", "").replace("_", " ").title() for k in ci_keys]
    means = [bootstrap[k]["mean"] * 100.0 for k in ci_keys]
    lowers = [bootstrap[k]["ci_lower"] * 100.0 for k in ci_keys]
    uppers = [bootstrap[k]["ci_upper"] * 100.0 for k in ci_keys]

    y_pos = range(len(ci_keys))
    ax2.errorbar(means, y_pos, xerr=[np.array(means) - np.array(lowers), np.array(uppers) - np.array(means)], fmt="o", color="#2c3e50", capsize=5, ecolor="#e74c3c")
    ax2.axvline(0.0, color="gray", linestyle="--", alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Accuracy Difference vs. Question-Aware SnapKV (%)")
    ax2.set_title("Paired 95% Bootstrap CIs (B=2,000)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    out_path = FIGURES_DIR / "task2_kv_cache_tradeoffs.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved {out_path}")


def plot_context_benchmark_results(ctx_path: Path):
    """Plot Single-Needle (NIAH K=1) and Multi-Needle (M-NIAH) Recall across policies."""
    if not ctx_path.exists():
        return
    with open(ctx_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    single = data.get("single_needle_niah", {})
    multi = data.get("multi_needle_mniah", data.get("option_b_mniah", {}).get("summary", {}))
    if not single and not multi:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    policies = ["fp16_reference", "sink_recent_fp16", "question_aware_fp16", "token_wise_int4", "channel_k_token_v_int4", "hybrid_fp16_int2"]
    labels = ["FP16 Ref", "Sink+Recent", "SnapKV", "Token INT4", "Channel INT4", "Hybrid INT2"]
    colors = ["#2c3e50", "#e74c3c", "#3498db", "#9b59b6", "#27ae60", "#f39c12"]

    # 1. Single-Needle NIAH
    ax1 = axes[0]
    single_scores = [single.get(p, {}).get("mean_accuracy", 0.0) * 100.0 for p in policies]
    bars1 = ax1.bar(labels, single_scores, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Exact Match Accuracy (%)")
    ax1.set_title("Single-Needle in a Haystack (NIAH K=1, 2k/4k Tokens)")
    ax1.set_ylim(0, 110)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax1.tick_params(axis="x", rotation=25)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=8.5)

    # 2. Multi-Needle M-NIAH
    ax2 = axes[1]
    multi_scores = [multi.get(p, {}).get("mean_needle_recall", 0.0) * 100.0 for p in policies]
    bars2 = ax2.bar(labels, multi_scores, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("Multi-Needle Recall (%)")
    ax2.set_title("Multi-Needle in a Haystack (M-NIAH K∈{2,4,8}, 2k/4k Tokens)")
    ax2.set_ylim(0, 110)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.tick_params(axis="x", rotation=25)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=8.5)

    out_path = FIGURES_DIR / "task2_context_mniah_recall.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved {out_path}")


def main():
    results_path = RESULTS_DIR / "all_results.json"
    if not results_path.exists():
        print(f"No results file found at {results_path}. Run Task 2 first.")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "part_a" in data:
        plot_attention_operators(data["part_a"])
    if "part_b" in data:
        plot_mqar_results(data["part_b"])
    if "part_c" in data:
        plot_kv_results(data["part_c"])

    ctx_file = RESULTS_DIR / "context_benchmarks.json"
    plot_context_benchmark_results(ctx_file)

    print("✓ All Task 2 figures generated successfully.")


if __name__ == "__main__":
    main()
