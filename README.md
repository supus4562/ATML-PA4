# ATML Programming Assignment 4 (Fall 2026)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers 4.57.1](https://img.shields.io/badge/transformers-4.57.1-yellow.svg)](https://huggingface.co/docs/transformers)
[![Modal](https://img.shields.io/badge/cloud-Modal%20Serverless%20GPU-green.svg)](https://modal.com)

Comprehensive implementation, 100% serverless cloud GPU benchmarking, and empirical analysis for **ATML Programming Assignment 4: Model Compression, Efficient Attention, and Context Memory**.

---

## 📁 Repository Structure

```text
├── task1/                     # Task 1: Model Compression Suite
│   ├── baseline.py            # Baseline evaluation (WikiText-2, C4, HellaSwag acc_norm)
│   ├── pruning.py             # P1 (Magnitude), P2 (Wanda), 2:4 Structured Sparsity
│   ├── non_uniform.py         # P3 (Activation-norm sensitivity + bisection allocation)
│   ├── quantization.py        # Q1 (Uniform PTQ), Q2 (AWQ grid search + scaling)
│   ├── mixed_precision.py     # Q3 (Hessian/MSE proxy sensitivity + integer knapsack)
│   └── runner.py              # Task 1 orchestrator and state manager
├── task2/                     # Task 2: Efficient Attention & Context Memory Suite
│   ├── attention_ops.py       # Part A: Dense, Tiled Online-Softmax, and SDPA operators
│   ├── mqar_models.py         # Part B: Dense, Sliding-Window, Linear, Dilated, DGLA
│   ├── mqar_train.py          # Part B: Multi-seed training (6304, 6305, 6306) and distance eval
│   ├── kv_cache_models.py     # Part C: Matched INT4 budget math, byte accounting
│   ├── kv_quantizers.py       # Part C: Vectorized Token-wise and Channel-K quantizers
│   ├── kv_policies.py         # Part C: Sink+Recent, SnapKV, Token INT4, Channel INT4, Hybrid
│   ├── kv_eval.py             # Part C: 180 factorial retrieval evaluation with cached prefill
│   └── eval_context_benchmark.py # Extension: Mini-RULER Suite (NIAH, M-NIAH, Variable Tracking)
├── data/                      # Dataset splits and locked evaluation manifests
│   ├── local_source_revisions.json # Frozen upstream Hugging Face commit SHAs
│   ├── public_locked/         # C4, WikiText-2, HellaSwag, GSM8K locked evaluation rows
│   ├── kv_retrieval/          # 180 factorial retrieval prompt cases (2k & 4k tokens)
│   ├── mqar/                  # 9,216 frozen test and 2,304 development MQAR sequences
│   └── short_form/            # 360 short-form instruction evaluation prompts
├── docs/                      # Scientific documentation and qualitative case studies
│   ├── task1_qualitative_analysis.md # HellaSwag prompt-level comparisons across P1-P3, Q1-Q3
│   ├── task2_qualitative_analysis.md # Mechanistic case studies (sliding window, SnapKV, INT2)
│   ├── DATA_FORMATS.md        # JSONL schemas and column definitions
│   └── REPRODUCIBILITY.md     # Environmental and experimental constraints
├── figures/                   # Publication-ready figures (PNG & vector PDF)
│   ├── task1_sparsity_tradeoff.png
│   ├── task1_quantization_tradeoff.png
│   ├── task1_bootstrap_ci.png
│   ├── task2_attention_operators.png
│   ├── task2_mqar_learning_curves.png
│   ├── task2_mqar_distance_heatmap.png
│   ├── task2_kv_cache_tradeoffs.png
│   └── task2_context_mniah_recall.png
├── runs/                      # Machine-readable JSON metrics and bootstrap intervals
│   ├── task1_results/         # all_results.json, bootstrap_ci.json, sensitivity profiles
│   └── task2_results/         # all_results.json, context_benchmarks.json, state.json
├── scripts/                   # Evaluation, verification, and plotting utilities
│   ├── check_setup.py         # Public data and revision lock verification
│   ├── verify_checksums.py    # Byte-exact SHA256 integrity verification
│   ├── validate_assets.py     # Structural invariant checks on MQAR and KV cases
│   ├── plot_task1_figures.py  # Generates all Task 1 publication figures
│   ├── plot_task2_figures.py  # Generates all Task 2 publication figures
│   ├── qualitative_evaluation.py # Generates Task 1 qualitative report
│   └── qualitative_task2.py   # Generates Task 2 qualitative report
├── tests/                     # Unit test suite
│   └── test_datasets.py       # Determinism and isolation tests
├── modal_task1.py             # Modal cloud orchestrator for Task 1 (NVIDIA L4 / A10G)
├── modal_task2.py             # Modal cloud orchestrator for Task 2 (NVIDIA A10G)
├── CHECKLIST.md               # Master requirements checklist and mental map
├── SHA256SUMS.json            # Checksum manifest
├── requirements.txt           # Environment dependencies
└── pyproject.toml             # Project metadata and tooling config
```

---

## 🚀 Quick Start & Environment Setup

### 1. Local Environment
Install dependencies in a virtual environment:
```bash
pip install -r requirements.txt
```

Verify data integrity and upstream source snapshots:
```bash
python scripts/check_setup.py
python scripts/verify_checksums.py
python scripts/validate_assets.py
python -m unittest discover -s tests -v
```

### 2. 100% Serverless Cloud Execution (Modal GPU)
All heavy compute (model forward passes, pruning, AWQ search, MQAR training, and KV compression) is orchestrated via [Modal](https://modal.com) serverless cloud containers:
```bash
# Configure Modal credentials
modal setup

# Execute complete Task 1 suite on NVIDIA L4 ($0.80/hr)
modal run modal_task1.py --run-all --gpu L4

# Execute complete Task 2 suite on NVIDIA A10G
modal run modal_task2.py --run-all --gpu A10G
```

---

## 🔬 Task 1: Model Compression (Pruning & Quantization)

Evaluates post-training weight compression on `Qwen/Qwen2.5-1.5B` base across WikiText-2, C4 validation, and zero-shot HellaSwag:

### Key Methods
* **P1 (Magnitude Pruning):** Unstructured per-output-row pruning ($S_{ij} = |W_{ij}|$) at $s \in \{0.25, 0.50, 0.75\}$.
* **P2 (Wanda Pruning):** Activation-aware pruning ($S_{ij} = |W_{ij}| \cdot \|X_{j,:}\|_2$) using calibration activations at $s \in \{0.25, 0.50, 0.75\}$.
* **P3 (Non-Uniform Pruning):** Activation-norm sensitivity profiling with bisection root-finding under exact $50\%$ and $75\%$ global weight budgets ($0.10 \le s_\ell \le 0.85$).
* **Q1 (Uniform PTQ):** Group-128 asymmetric weight-only quantization at $b \in \{8, 4, 3\}$.
* **Q2 (AWQ):** Activation-aware grid search for per-channel scaling factors $s = S_X^\alpha$, protecting outlier channels.
* **Q3 (Mixed-Precision Quantization):** Knapsack allocation across $\{2, 3, 4, 8\}$ bits matching uniform INT4 and INT3 budgets.
* **2:4 Structured Sparsity:** Evaluates 2-out-of-4 contiguous Wanda pruning at $50\%$ sparsity.

### Reproducing Task 1 Figures & Qualitative Analysis
```bash
python scripts/plot_task1_figures.py
python scripts/qualitative_evaluation.py
```

---

## ⚡ Task 2: Efficient Attention & Context Memory

### Part A: Attention Operators
* Implements **Dense Causal Attention**, **Tiled Online-Softmax Causal Attention**, and **PyTorch SDPA** (`flash` backend).
* Verifies strict causality ($0.0$ max difference under future perturbation) and numerical stability under extreme unscaled dynamic ranges $[-10^4, 10^4]$.
* Measures memory scaling up to $L = 8,192$, demonstrating that online tiling reduces activation footprint from $>20\text{ GB}$ (dense OOM) down to $138\text{ MB}$.

### Part B: Multi-Query Associative Recall (MQAR)
* Trains 2-layer Transformers from scratch across 3 distinct seeds (`6304`, `6305`, `6306`) for 4,500 updates on MQAR ($T=256, K \in \{4, 8, 16\}$).
* Compares **Dense**, **Sliding Window ($W=64$)**, **Linear Attention ($\text{ELU}+1$)**, **Mixed Local/Dilated Sparse**, and **Decay-Gated Linear Attention (DGLA)**.
* Proves the analytical $2(W-1) = 126$ token receptive field cutoff of sliding-window attention on far queries ($[128, 256)$), where accuracy collapses to the $1/64 = 1.56\%$ random guessing floor.

### Part C: Pretrained KV Cache Compression
Evaluated on `Qwen/Qwen2.5-1.5B-Instruct` across 180 factorial retrieval cases ($L \in \{2048, 4096\}$, evidence at start/middle/end, distractors $0, 2, 8$) under a strictly matched 4-bit serialized byte budget $M_b(L) \approx 30.7\text{ MB}$:

| Policy | Exact Match Acc | Conditional Acc | Mean Answer NLL | Mean Logit KL | Retained Evidence | Packed MB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Question-Aware SnapKV** | **$95.0\%$** | **$98.8\%$** | **$0.010$** | **$0.024$** | **$100.0\%$** | **$30.7\text{ MB}$** |
| **FP16 Reference** | $91.1\%$ | $100.0\%$ | $0.031$ | $0.000$ | $100.0\%$ | $107.3\text{ MB}$ |
| **Token-wise INT4** | $57.2\%$ | $61.0\%$ | $0.685$ | $0.629$ | $100.0\%$ | $30.2\text{ MB}$ |
| **Sink + Recent FP16** | $32.8\%$ | $34.1\%$ | $1.790$ | $1.838$ | $33.3\%$ | $30.7\text{ MB}$ |
| **Channel-K / Token-V INT4** | $6.1\%$ | $6.7\%$ | $5.870$ | $5.780$ | $100.0\%$ | $31.8\text{ MB}$ |
| **Hybrid FP16 + INT2** | $3.9\%$ | $4.3\%$ | $1.850$ | $1.750$ | $100.0\%$ | $30.5\text{ MB}$ |

### Mini-RULER Context Memory Suite
* **Single-Needle NIAH ($K=1$):** FP16 Reference ($100.0\%$), SnapKV ($100.0\%$), Sink+Recent ($33.3\%$).
* **Multi-Needle M-NIAH ($K \in \{2, 4, 8\}$):** SnapKV ($62.5\%$), FP16 Reference ($52.1\%$), Token-wise INT4 ($18.8\%$).
* **Variable Tracking (Multi-Hop Causal Reasoning):** FP16 Reference ($25.0\%$), Compressed policies ($0.0\%$). Reveals the fundamental failure mode of single-query attention eviction for multi-hop coreference resolution.

### Reproducing Task 2 Figures & Qualitative Analysis
```bash
python scripts/plot_task2_figures.py
python scripts/qualitative_task2.py
```

---

## 📜 Attributions & References

We gratefully acknowledge the following research and open-source foundations:
* **Transformers & Accelerate:** [Hugging Face](https://github.com/huggingface/transformers) (version `4.57.1` pinned for cache-editing interfaces).
* **Modal Serverless Cloud:** [Modal Labs](https://modal.com) for GPU container infrastructure.
* **FlashAttention / Online Tiling:** Dao et al., *"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"*, NeurIPS 2022.
* **Wanda Pruning:** Sun et al., *"A Simple and Effective Pruning Approach for Large Language Models"*, ICLR 2024.
* **AWQ:** Lin et al., *"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"*, MLSys 2024.
* **SnapKV:** Li et al., *"SnapKV: LLM Knows What You Are Looking for Before Generation"*, 2024.
* **KIVI:** Liu et al., *"KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"*, ICML 2024.
* **RULER Benchmark:** Hsieh et al., *"RULER: What's the Real Context Size of Your Long-Context Language Models?"*, 2024.
* **Qwen2.5 Models:** Alibaba Cloud / Qwen Team, *"Qwen2.5 Technical Report"*, 2024.
