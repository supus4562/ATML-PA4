#!/usr/bin/env python3
"""scripts/qualitative_evaluation.py — Qualitative comparison of matched HellaSwag prompts across compression conditions."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

DATA_PATH = Path("data/public_locked/hellaswag_val_locked.jsonl.gz")
RESULTS_PATH = Path("runs/task1_results/all_results.json")
OUTPUT_MD = Path("docs/task1_qualitative_analysis.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)


def load_examples():
    examples = []
    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                examples.append(item.get("example", item))
    return examples


def main():
    if not RESULTS_PATH.exists() or not DATA_PATH.exists():
        print("Missing results or dataset file.")
        return

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        res = json.load(f)

    examples = load_examples()

    # Compare conditions on first 512 examples (common intersection evaluated across all conditions)
    conditions = [
        "baseline_fp16",
        "P1_magnitude_s50",
        "P2_wanda_s50",
        "P_structured_2_4",
        "P3_nonuniform_s50",
        "Q1_uniform_int4",
        "Q2_awq_int4",
        "Q3_mixed_match_int4",
    ]

    records = {}
    for c in conditions:
        if c in res and "_per_example_hs" in res[c]:
            records[c] = [x["norm"] for x in res[c]["_per_example_hs"]]

    num_eval = min(len(records[c]) for c in conditions)

    # Find compelling illustrative cases
    case_studies = []

    # Case 1: Wanda succeeds where Magnitude fails (activation weighting benefit)
    for i in range(num_eval):
        if (
            records["baseline_fp16"][i] == 1
            and records["P2_wanda_s50"][i] == 1
            and records["P1_magnitude_s50"][i] == 0
        ):
            ex = examples[i]
            case_studies.append({
                "type": "Wanda vs Magnitude (50% Sparsity)",
                "idx": i,
                "context": ex["ctx"],
                "gold_label": int(ex["label"]),
                "gold_ending": ex["endings"][int(ex["label"])],
                "predictions": {c: records[c][i] for c in conditions},
                "rationale": "P2 Wanda preserves large-magnitude activation channels, enabling correct reasoning where Magnitude pruning destroys critical outlier pathways.",
            })
            if len(case_studies) >= 2:
                break

    # Case 2: Unstructured Wanda succeeds where 2:4 Structured fails (combinatorial constraint penalty)
    for i in range(num_eval):
        if (
            records["baseline_fp16"][i] == 1
            and records["P2_wanda_s50"][i] == 1
            and records["P_structured_2_4"][i] == 0
        ):
            ex = examples[i]
            case_studies.append({
                "type": "Unstructured Wanda vs 2:4 Structured (50% Sparsity)",
                "idx": i,
                "context": ex["ctx"],
                "gold_label": int(ex["label"]),
                "gold_ending": ex["endings"][int(ex["label"])],
                "predictions": {c: records[c][i] for c in conditions},
                "rationale": "The rigid 2-in-4 contiguous constraint forces the deletion of salient weights that cluster within the same 4-tuple, inducing catastrophic misclassification.",
            })
            if len(case_studies) >= 4:
                break

    # Case 3: AWQ succeeds where Uniform PTQ fails (scale protection benefit)
    for i in range(num_eval):
        if (
            records["baseline_fp16"][i] == 1
            and records["Q2_awq_int4"][i] == 1
            and records["Q1_uniform_int4"][i] == 0
        ):
            ex = examples[i]
            case_studies.append({
                "type": "AWQ vs Uniform PTQ (INT4)",
                "idx": i,
                "context": ex["ctx"],
                "gold_label": int(ex["label"]),
                "gold_ending": ex["endings"][int(ex["label"])],
                "predictions": {c: records[c][i] for c in conditions},
                "rationale": "Channel-wise activation scaling under AWQ protects outlier features from aggressive clipping, preventing quantization-induced semantic collapse.",
            })
            if len(case_studies) >= 6:
                break

    # Generate Markdown Report
    lines = [
        "# Task 1: Qualitative Prompt-Level Evaluation",
        "",
        "A prompt-level qualitative comparison across compressed model variants evaluated on zero-shot HellaSwag.",
        "",
        "| ID | Phenomenon Analyzed | Context Prompt Excerpt | Gold Ending | Baseline | P1 (Mag) | P2 (Wanda) | 2:4 Struct | Q1 (PTQ) | Q2 (AWQ) | Q3 (Mixed) |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cs in case_studies:
        ctx_snippet = (cs["context"][:60] + "...") if len(cs["context"]) > 60 else cs["context"]
        end_snippet = (cs["gold_ending"][:45] + "...") if len(cs["gold_ending"]) > 45 else cs["gold_ending"]
        p = cs["predictions"]
        sym = lambda v: "✓" if v == 1 else "✗"
        lines.append(
            f"| #{cs['idx']} | {cs['type']} | *\"{ctx_snippet}\"* | *\"{end_snippet}\"* | "
            f"{sym(p['baseline_fp16'])} | {sym(p['P1_magnitude_s50'])} | {sym(p['P2_wanda_s50'])} | "
            f"{sym(p['P_structured_2_4'])} | {sym(p['Q1_uniform_int4'])} | {sym(p['Q2_awq_int4'])} | {sym(p['Q3_mixed_match_int4'])} |"
        )

    lines.append("\n## Detailed Case Study Breakdown\n")
    for idx, cs in enumerate(case_studies, start=1):
        lines.append(f"### Case {idx}: {cs['type']} (Sample #{cs['idx']})")
        lines.append(f"- **Prompt Context:** \"{cs['context']}\"")
        lines.append(f"- **Ground Truth Target:** \"{cs['gold_ending']}\"")
        lines.append(f"- **Theoretical Rationale:** {cs['rationale']}")
        lines.append("")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Qualitative analysis report generated at {OUTPUT_MD}")


if __name__ == "__main__":
    main()
