# ATML Programming Assignment 4 — Master Requirements & Mental Map Checklist

This document serves as the single source of truth and comprehensive checklist for **ATML Programming Assignment 4 (Fall 2026)**. It covers administrative rules, step-by-step instructions, required empirical evidence, and hard constraints across all three tasks.

---

## 1. Global Administrative & Experimental Rules

### Administrative & Report Constraints
- [ ] **Due Date:** 30 November 2026 via LMS.
- [ ] **Individual Submission:** Work must be strictly individual. No collaboration or sharing of code, experiments, or text.
- [ ] **Report Page Limit:** Exactly **8 pages of main content** in the course NeurIPS style. References and supplementary appendix do not count toward the 8 pages; however, all primary required evidence must reside in the main report.
- [ ] **Public GitHub Repository:** Code must be hosted publicly. The GitHub URL must appear at the end of the abstract.
- [ ] **AI Usage Policy:** **Zero AI-generated text in the report.** Interpretation, analysis, and writing must be entirely student-authored. LLM coding assistance for implementations is permitted, but all code must be understood and materially reused code must be attributed in `README.md`.

### Environment & Reproducibility Constraints
- [ ] **Pinned Libraries:** `transformers==4.57.1` and `huggingface_hub==0.36.2` (mandatory for cache-editing interfaces).
- [ ] **Master Reference Seed:** `6304` is the default seed. Additional seeds (`6305`, `6306`) are strictly reserved for specified multi-seed training runs.
- [ ] **Data Provenance:** Keep and commit `data/local_source_revisions.json` to freeze resolved Hugging Face commit hashes.
- [ ] **Benchmarking Protocol:**
  - [ ] Warm up all timed code paths prior to measurement.
  - [ ] Enclose all GPU timings with `torch.cuda.synchronize()`.
  - [ ] Report measurement units and number of repetitions ($\ge 5$ or $\ge 10$ warmed-up runs).
  - [ ] Separate calibration/preprocessing, prefill, decode, and postprocessing.
  - [ ] Distinguish **logical/packed storage** from **physical GPU VRAM allocation**.
  - [ ] Distinguish **theoretical arithmetic operations** from **measured hardware speedup**.
- [ ] **Evaluation Integrity:**
  - [ ] ❌ **NEVER** use test sets (WikiText-2, C4 val, HellaSwag test, MQAR test, 120 speculative test prompts) for calibration or hyperparameter selection.
  - [ ] Report all conditions, including negative results, unlearned runs, or OOM crashes.

---

## 2. Task 1: Model Compression (Pruning & Quantization)

### Model & Setup
- [ ] **Base Model:** Pretrained `Qwen/Qwen2.5-1.5B` base model only (**NOT** the Instruct version).
- [ ] **Eligible Weight Projections (7 per block):**
  - Self-attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - MLP: `gate_proj`, `up_proj`, `down_proj`
- [x] **Excluded from Compression:** Embeddings, LayerNorm/RMSNorm, biases, and `lm_head` remain uncompressed FP16. Count tied/shared weights once.
- [x] **Zero Fine-Tuning:** Post-training compression only; weights are never updated via gradient descent.
- [x] **Calibration & Development Splits:**
  - Calibration: 128 sequences $\times$ 512 tokens from C4 (`c4_calibration_locked.jsonl.gz`, seed `6304`).
  - Development: 32 sequences $\times$ 512 tokens from C4 (`c4_development_locked.jsonl.gz`) for hyperparameter tuning.

### Step-by-Step Instructions
- [x] **Step 1: Baseline**
  - Load FP16 base model; evaluate WikiText-2 and C4 perplexities and HellaSwag accuracy.
  - Measure uncompressed packed/dense bytes, peak VRAM, prefill latency (512, 2,048 tokens), and 64-token decode latency.
- [x] **Step 2: P1 — Magnitude Pruning**
  - Apply unstructured per-output-row pruning ($S_{ij} = |W_{ij}|$) at target sparsities $s \in \{0.25, 0.50, 0.75\}$.
  - Restore original FP16 weights between sparsities.
- [x] **Step 3: P2 — Wanda Pruning**
  - Apply activation-aware pruning ($S_{ij} = |W_{ij}| \cdot \|X_{j,:}\|_2$) using calibration activations at $s \in \{0.25, 0.50, 0.75\}$.
  - Save per-projection score distributions for P3.
- [x] **Step 4: P3 — Non-Uniform Pruning**
  - Implement a sensitivity measure (e.g., Taylor expansion, activation norm, block influence) and an allocation rule.
  - Evaluate at global sparsities **50% and 75%**, exactly matching total zeroed weights in P1/P2.
  - Enforce bounds: $0.10 \le s_\ell \le 0.85$ per allocation unit.
- [x] **Step 5: Q1 — Uniform Post-Training Quantization (PTQ)**
  - Group-wise asymmetric weight-only PTQ at bit widths $b \in \{8, 4, 3\}$ with group size 128 along input columns.
- [x] **Step 6: Q2 — Activation-Aware Weight Quantization (AWQ)**
  - Implement activation-dependent per-channel scaling and inverse compensation at $b \in \{8, 4, 3\}$, group size 128.
  - Search channel scales on calibration activations only; fold into preceding linear layers where valid or store FP16 scales.
- [x] **Step 7: Q3 — Mixed-Precision Quantization**
  - Design sensitivity measure and allocation rule assigning bit widths from $\{2, 3, 4, 8\}$ using AWQ quantizer.
  - Construct two allocations matching total packed-byte budgets of uniform Q2-INT4 and Q2-INT3.
- [x] **Step 8: Structured 2:4 Sparsity Diagnostic**
  - At 50% sparsity, compare a 2:4 structured mask (2 highest-ranked weights in each 4 contiguous input weights) vs. 50% unstructured Wanda.
- [x] **Step 9: Trade-Off Analysis**
  - Plot quality vs. global sparsity and quality vs. actual packed bytes.

### Required Evidence
- [x] Perplexity on WikiText-2 (50k scored tokens) and C4 validation (50k scored tokens): report $N$, mean NLL, and PPL.
- [x] Zero-shot HellaSwag: Raw accuracy and character-length-normalized accuracy (`acc_norm`). Full 2,000 examples for baseline, 50% pruning conditions, and INT4/INT3 conditions; first 512 examples diagnostic subset for others.
- [x] Paired 95% bootstrap confidence intervals for $P_3 - P_2$ and $Q_3 - Q_2$ differences.
- [x] Sensitivity visualizations across layers/projections for P3 and Q3.
- [x] Final sparsity and bit-allocation maps.
- [x] Resource accounting: achieved sparsity, packed storage in bytes (with metadata), peak GPU VRAM, prefill/decode latency, calibration wall-clock time.
- [x] Qualitative evaluation: matched HellaSwag prompts comparing predictions across configurations.

### Hard Constraints & Don'ts
- [ ] ❌ **DO NOT** count uncompressed components (embeddings, heads) toward the sparsity percentage of P1/P2/P3.
- [ ] ❌ **DO NOT** claim dense-with-zeros or fake-quantized FP16 tensors represent physical GPU speedups or VRAM savings without actual sparse/int kernels.
- [ ] ❌ **DO NOT** normalize HellaSwag by token count (must use character-length normalization `acc_norm`).
- [ ] ❌ **DO NOT** score the first token of independent sequences in the perplexity denominator ($N$).

---

## 3. Task 2: Efficient Attention & Context Memory

### Part A: Attention Operators
- [ ] **Implement 3 Operators:**
  1. Dense causal attention ($QK^\top / \sqrt{d}$, causal mask, stable softmax, multiply $V$).
  2. Tiled online-softmax causal attention (blockwise forward pass without full $L \times L$ matrix).
  3. PyTorch `torch.nn.functional.scaled_dot_product_attention` (SDPA).
- [ ] **Test Grid:** Synthetic inputs with $B = 2, H = 4, d_h = 64$; lengths $L \in \{256, 512, 1024, 2048, 4096, 8192\}$; FP16 and FP32.
- [ ] **Causality Verification:** Perturb future keys/values and verify past outputs remain strictly invariant.
- [ ] **Evidence:** Output error ($\max |y - y_{\text{ref}}|$) and gradient error ($\max |g - g_{\text{ref}}|$) vs. FP32 dense reference; forward latency vs. full forward+backward latency; peak VRAM; recorded SDPA backend.
- [ ] ❌ **DO NOT** claim Python-tiled autograd represents a fused FlashAttention kernel.

### Part B: Associative Recall (MQAR)
- [ ] **Model Architecture (from scratch):**
  - 2 decoder blocks, hidden dimension 128, 4 heads ($d_h = 32$), MLP $4\times$, vocab 256.
  - Learned token embeddings, sinusoidal positional embeddings.
- [ ] **Training Protocol:**
  - Optimizer: AdamW (peak LR $8 \times 10^{-4}$, weight decay 0.01).
  - Schedule: 32 sequences/batch, 4,500 updates, 200 warmup, linear decay to 15%, grad norm clip 1.0.
  - Multi-seed requirement: Run seeds **`6304`**, **`6305`**, **`6306`** for all required models.
- [ ] **Data Generator:** MQAR at $T = 256, K \in \{4, 8, 16\}$.
  - Regimes: Near (distance $< 64$), Far (distance in $[128, 250]$), Mixed (all feasible distances).
- [ ] **Models Trained:**
  - Dense attention: Mixed regime at $K \in \{4, 8, 16\}$ + Near-only at $K = 16$.
  - Sliding Window ($W = 64$): Near-only at $K \in \{4, 8, 16\}$ + Mixed at $K = 16$.
  - Linear Attention (causal ELU+1): Mixed regime at $K \in \{4, 8, 16\}$.
  - Mixed Local/Dilated Sparse: Mixed regime at $K \in \{4, 16\}$ (2 heads attend to preceding 64; 2 heads attend to offsets $\{0, 4, 8, \dots, 252\}$).
  - Test-time mask control: Apply $W = 64$ mask at test-time to learned dense model.
  - Controlled Intervention: One motivated architectural/training modification at $K = 16$ across all 3 seeds (within 5% baseline parameters).
- [ ] **Forward-Only Sequence Sweep:** Fixed random inputs at $T \in \{256, 1024, 4096, 8192\}$.
- [ ] **Evidence:** Learning curves; test accuracy on 1,024 test sequences across distance bins $[0, 32), [32, 64), [64, 128), [128, 256)$; comparison against label chance ($1/64$) and in-context chance ($1/K$).
- [ ] ❌ **DO NOT** use next-token loss (MQAR is supervised query-position classification).
- [ ] ❌ **DO NOT** use a single fixed offset of 128 for far queries (must span $[128, 250]$).

### Part C: Pretrained KV Memory Compression
- [ ] **Model:** Pretrained `Qwen/Qwen2.5-1.5B-Instruct` in FP16, greedy decoding.
- [ ] **Data:** 180 frozen long-context retrieval prompts (`kv_retrieval_cases.jsonl.gz`) across 2,048 & 4,096 tokens, gold evidence at start/middle/end, 0/2/8 distractors.
- [ ] **Case-Specific Matched Budget:** Exact serialized byte size of **INT4 channel-wise-K / token-wise-V cache** with group size 64:
  $$M_b(L) = 2 n_\ell H_{\text{kv}} d_h \left[ \left\lfloor \frac{L}{64} \right\rfloor \left( \frac{64 \cdot b}{8} + 4 \right) + 2(L \bmod 64) \right]$$
- [ ] **5 Policies Evaluated Under Identical Budget:**
  1. **Sink + Recent FP16 Eviction:** 8 sink tokens + 64 recent tokens + fill budget with most recent positions.
  2. **Question-Aware FP16 Eviction:** 8 sink tokens + 64 recent tokens + rank remaining positions using question-suffix attention.
  3. **Token-wise INT4:** Both K and V grouped along hidden dimension per token (group 64).
  4. **Channel-wise K / Token-wise V INT4:** K grouped along tokens per channel; V grouped along channels per token (group 64).
  5. **Hybrid Precision & Selection:** 8 sink + 64 recent in FP16 + top question-aware positions in FP16 + compress remaining positions in INT2 channel-K/token-V.
- [ ] **Evidence:** Full-precision split-prefill/decode equivalence sanity check; matched-budget table (exact-match accuracy, answer NLL, logit KL from FP16, retained evidence fraction, packed bytes, peak VRAM); paired bootstrap intervals ($\ge 2,000$ resamples); independent encode/decode benchmarks at batch 1 and 8 (lengths 2k and 4k).
- [ ] ❌ **DO NOT** give gold evidence character coordinates to retention algorithms.
- [ ] ❌ **DO NOT** lose original absolute rotary position IDs (RoPE) when evicting tokens.

---

## 4. Task 3: Policy Distillation & Speculative Decoding

### Models & Setup
- [ ] **Target / Teacher Model:** Frozen `Qwen/Qwen2.5-1.5B-Instruct` (FP16).
- [ ] **Draft / Student Model:** Frozen `Qwen/Qwen2.5-0.5B-Instruct` backbone (FP16) + trainable LoRA adapters.
- [ ] **LoRA Configuration:** $r = 8, \alpha = 16, \text{dropout} = 0$; applied to `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- [ ] **Prompt Formatting:** Identical non-thinking assistant chat template and fixed EOS handling.

### Step-by-Step Instructions
- [ ] **Step 1: Implement & Validate Exact Speculative Sampling**
  - Implement proposal generation, single verification pass, stochastic acceptance rule $a = \min(1, p/q)$, positive residual sampling $r = [p - q]^+ / \sum [p - q]^+$, bonus token, and cache rollback/advancement.
  - **Markov Toy Validation:** 3-symbol Markov chain, 100,000 sampled sequences of length 3.
  - Pearson $\chi^2$ goodness-of-fit test must **not reject** correct sampler ($p \ge 0.01$).
  - Negative control: replace residual with $p$; $\chi^2$ test must **reject** ($p < 0.01$).
- [ ] **Step 2: Pretrained Draft Baseline**
  - Benchmark target-only vs. speculative decoding with original 0.5B draft at $\gamma \in \{2, 4\}$.
- [ ] **Step 3: Off-Policy Distillation**
  - Prepare fixed teacher rollouts from 500 GSM8K + 300 short-form training prompts (temp 1, top-p 1).
  - Train student with forward token KL: $\mathcal{L}_{\text{KD}} = \frac{1}{|H|} \sum_{h \in H} D_{\text{KL}}(p_T(\cdot|h) \parallel q_\phi(\cdot|h))$.
  - Fixed quota: **40,000 teacher-scored tokens** (30k math, 10k short-form) for seeds **`6304`** and **`6305`**.
  - AdamW LR $3 \times 10^{-4}$, weight decay 0.01, 1 update per 256 scored tokens.
- [ ] **Step 4: On-Policy Distillation**
  - Same prompts, same 40,000-token quota, same seeds `6304` and `6305`.
  - Rollouts generated dynamically by current student policy, evaluated by teacher, trained with same forward KL.
- [ ] **Step 5: Common Prefix Distributional Analysis**
  - Evaluate draft models on 4 prefix banks: (i) Teacher prefixes, (ii) Original student prefixes, (iii) Exact speculative prefixes ($\gamma = 4$), and (iv) Each trained student's own rollout prefixes.
  - Compute: Forward KL, Reverse KL, Total Variation (TV), Greedy next-token agreement, and Expected one-token acceptance ($1 - \text{TV}$).
- [ ] **Step 6: Real Speculative Execution with Merged Adapters**
  - **Merge LoRA adapters** into fresh 0.5B backbones; verify zero logit discrepancy.
  - Benchmark 120 held-out prompts (60 GSM8K + 60 short-form) at $\gamma \in \{2, 4\}$.
  - Measure throughput (tokens/sec), speedup ratio $S_\gamma$, acceptance rate $\alpha$, and 4-part per-round latency breakdown:
    1. Draft proposal time
    2. Target verification time
    3. Acceptance / residual sampling time
    4. Cache rollback / advancement time
  - Compare measured speedup $S_\gamma$ against theoretical cost model $\hat{S}_\gamma = \frac{1 + \sum_{j=1}^\gamma \alpha^j}{\gamma c + v_\gamma}$ and compute break-even $\alpha$.
- [ ] **Step 7: Workload Breakdown & Deployment Economics**
  - Break down results by domain: Mathematics vs. Familiar extraction vs. Shifted extraction.
  - Measure standalone student task accuracy (512-token cap for math, 32 for short-form).
  - Calculate upfront training cost amortized over number of inference requests.

### Required Evidence
- [ ] Sampler correctness proof: 100,000-trial $\chi^2$ results ($p$-values for correct sampler and rejected negative control).
- [ ] Training curves: Loss vs. teacher-scored tokens, training wall-clock time, and seed variance.
- [ ] Prefix bank comparison table: Forward KL, Reverse KL, TV, Greedy agreement, and $1 - \text{TV}$ across prefix banks and domains.
- [ ] Speculative performance table: Empirical acceptance rate, mean accepted prefix length, target passes per token, throughput, latency, and peak GPU memory for $\gamma = 2, 4$.
- [ ] Paired 95% bootstrap confidence intervals for throughput and acceptance differences.
- [ ] Time decomposition chart / table: Draft, Verify, Sample, and Cache rollback components.
- [ ] Cost model analysis: Values of $c$, $v_\gamma$, $\hat{S}_\gamma$, measured $S_\gamma$, and break-even $\alpha$.
- [ ] Qualitative examples of prompt continuations showing agreements/disagreements.

### Hard Constraints & Don'ts
- [ ] ❌ **DO NOT** replace exact speculative sampling with greedy token matching (must implement true stochastic acceptance and residual sampling).
- [ ] ❌ **DO NOT** treat speculative prefix banks as a distinct target distribution (they follow target law by construction).
- [ ] ❌ **DO NOT** run primary latency benchmarks with unmerged PEFT/LoRA wrappers (merging into the base model is mandatory to eliminate wrapper overhead).
- [ ] ❌ **DO NOT** use the 64-token speed test outputs to evaluate math correctness (use the 512-token cap evaluation).
- [ ] ❌ **DO NOT** differentiate through student generation during on-policy training.

---

## 5. Deliverables & Pre-Submission Audit

### Repository Cleanliness & Artifacts
- [ ] Clean directory structure containing all runnable code for Tasks 1, 2, and 3.
- [ ] `requirements.txt` / `pyproject.toml` specifying exact environments.
- [ ] `scripts/setup_data.py` runnable with `--public` and `--prefix-banks`.
- [ ] `data/local_source_revisions.json` committed.
- [ ] Machine-readable JSON/CSV files containing all numerical results and per-condition logs.
- [ ] `README.md` with explicit instructions to reproduce every table and figure.
- [ ] Attribution section in `README.md` crediting all external libraries and code references.
- [ ] ❌ **DO NOT** commit huge `.pt` model weights or raw full datasets to GitHub.

### Final Report Quality
- [ ] Abstract contains the public GitHub repository URL.
- [ ] Report length is exactly 8 pages (excluding references and supplementary appendix).
- [ ] Answers all research questions using empirical evidence, compact tables, and clear figures with error bars / bootstrap intervals.
- [ ] Zero AI-generated text in the PDF report.
