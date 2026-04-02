# VQA Competition TOP-1 Plan (FINAL v3)

**Created:** 2026-04-02
**Revised:** 2026-04-02 (Consensus iteration 2 -- FINAL. All Architect + Critic fixes incorporated.)
**Deadline:** 2026-04-06 (4 days remaining)
**Objective:** Achieve TOP-1 accuracy on SSAFY 15th AI Challenge recycling image VQA

---

## RALPLAN-DR Summary

### Principles (5)

1. **Fix the loss signal first.** The baseline trains on the entire input sequence (~500-850 tokens) when only 1-2 answer tokens matter. Label masking is the single highest-ROI fix available, worth 5-15pp alone.
2. **Data is the biggest lever after loss.** The baseline uses 200/5073 train samples. Using all 5,073 train samples is the second-largest accuracy gain.
3. **Fit the largest viable model in 16GB VRAM.** Qwen2.5-VL-7B with 4-bit quantization fits (~6GB base + LoRA overhead). 7B >> 3B for Korean VQA understanding.
4. **Honest timelines prevent wasted effort.** Measured benchmark: 0.633 s/step for 3B at 384px. All training time estimates must be derived from this measurement, not wishes.
5. **Depth beats breadth on a single GPU.** Deeply tuning one model + log-prob inference > chasing ensemble diversity across 5 models sequentially.

### Decision Drivers (top 3)

1. **VRAM constraint (16GB):** Eliminates models >7B even with 4-bit quant. Rules out batch_size>2 for 7B. Makes efficient training (gradient checkpointing, mixed precision) mandatory.
2. **Time constraint (4 days, single GPU):** Measured 0.633 s/step for 3B@384px. 7B at 504px is ~2x slower (~1.3 s/step). A single 7B 3-epoch run takes ~5h. Entire budget is ~48h GPU time maximum. Must prioritize highest-ROI changes first.
3. **Dev data noise (only 20.2% have >=4/5 agreement):** At >=3/5 agreement, 74.3% of qualifying samples have only 3/5 agreement (42-75% label accuracy). Only 891 samples have >=4/5 agreement (~95%+ accuracy), and those have massive d-skew (45.8%) requiring answer-label stratification (464 usable after cap). High threshold is mandatory. A/B test required before adoption.

### Viable Options

**Option A: Qwen2.5-VL-7B-Instruct (Primary Recommendation)**
- Pros: Best Korean VQA capability in the size class; proven architecture; direct upgrade from baseline 3B; fits in 16GB with 4-bit quant (~6-7GB base weight)
- Cons: Slower training (~2x vs 3B); tighter VRAM margin means batch_size=1 only; ~5h per 3-epoch run
- Risk: If VRAM overflows, fall back to 3B with better training

**Option B: InternVL2.5-4B or InternVL3-2B**
- Pros: Strong multilingual VQA benchmarks; InternVL3-2B is very VRAM-friendly; native multi-image support
- Cons: Less proven on Korean specifically; different processor API requires significant code changes; InternVL2.5-4B quantized may not outperform Qwen 7B quantized
- Risk: API incompatibility could cost a full day of debugging
- **Invalidation rationale:** On a 4-day timeline with a single GPU, the code restructuring cost (~1 day) makes this option unviable as a primary. Kept only as a stretch diversity candidate.

**Option C: Qwen2.5-VL-3B-Instruct (Enhanced Baseline)**
- Pros: Already working; fast training (~2.5h per 3-epoch run); very VRAM-safe; can run batch_size=2+
- Cons: Fundamental capability ceiling lower than 7B; diminishing returns past 3-5 epochs on small data
- When to use: Fallback if 7B fails to load or train stably; ensemble diversity partner

**Decision: Option A (Qwen2.5-VL-7B) as primary, Option C (enhanced 3B) as fallback and ensemble partner. 2-model ensemble maximum.**

---

## Context

### Current Baseline Weaknesses (6 critical issues, ranked by impact)

| # | Issue | Impact | Fix |
|---|-------|--------|-----|
| 0 | **Labels train on entire sequence** -- `enc["labels"] = enc["input_ids"].clone()` wastes ~99.5% of loss on noise | ~5-15pp accuracy loss, slow convergence | Mask labels to -100 for all non-answer tokens |
| 1 | Only 200/5073 train samples used | ~75% of data wasted | Use all 5,073 samples |
| 2 | 1 epoch only | Severe underfitting | Train 2-3 epochs |
| 3 | IMAGE_SIZE=384x384 | Loses detail in 720x960+ images | Target 504 for both. 7B fallback chain: 504-->448-->384 if OOM |
| 4 | English system prompt for Korean task | Cross-lingual confusion | Switch to Korean prompt |
| 5 | `extract_choice()` defaults to "a" on parse failure | Silent wrong answers | Fix to use log-prob selection from the start |
| 6 | `peft` and `bitsandbytes` not in pyproject.toml | Build breaks on fresh env | `uv add peft bitsandbytes` |

### Data Analysis Summary

- **Train:** 5,073 samples, balanced ~1,250/class (a/b/c/d), 1 image per sample (no duplicates)
  - Question types: 34.5% counting / 65.5% material
- **Dev:** 4,413 samples, 5 noisy annotators each
  - Agreement=5: 0 samples (0.0%)
  - Agreement>=4: 891 samples (20.2%) -- HIGH confidence pseudo-labels (~95%+ accuracy)
    - Question types: 702 counting / 189 non-counting (78.8% / 21.2%)
    - Within non-counting: 77 material, 112 other
    - Answer distribution: a=180(20.2%), b=187(21.0%), c=116(13.0%), d=408(45.8%) -- **MASSIVE d-skew**
    - After answer-label stratification (cap each to min class c=116): 464 usable samples
    - After further question-type stratification: only 242 usable samples
  - Agreement>=3: 3,508 samples (79.5%) -- MEDIUM confidence, but 74.3% of these have only 3/5 agreement (42-75% label accuracy)
  - Agreement=2 or 1: 905 samples (20.5%) -- TOO NOISY, discard
  - Question types (overall): 69.5% counting / 30.5% material -- **INVERSE of train distribution**
- **Test:** 5,074 samples, no labels
- **Images:** 720-1600px wide, 720-1280px tall, all RGB

### Training Time Benchmarks (Measured)

| Configuration | s/step | Steps/epoch (4566 train) | Time/epoch | 3 epochs |
|---------------|--------|--------------------------|------------|----------|
| 3B @ 384px | 0.633 | 4566 | ~48min | ~2.4h |
| 3B @ 504px (est. 1.7x) | ~1.08 | 4566 | ~82min | ~4.1h |
| 7B @ 504px (est. 2x of 3B@504) | ~2.05 | 4566 | ~156min | ~7.8h |
| 7B @ 448px (est. 1.5x of 3B@504) | ~1.62 | 4566 | ~123min | ~6.2h |

**Total GPU budget: ~48h max (4 days). Realistic usable: ~36h (accounting for setup, debugging, inference).**

---

## Guardrails

### Must Have
- All code uses HuggingFace models only (no API calls)
- Every change measured against local validation split before submission
- Submission CSV format: `id,answer` with answers in {a,b,c,d}
- Model + training fits within 16GB VRAM
- At least one safe submission uploaded by end of Day 2
- Per-epoch checkpoint saving (`save_strategy="epoch"` or manual `model.save_pretrained()` per epoch)
- Label masking implemented before any training run
- `peft` and `bitsandbytes` added to pyproject.toml (Hour 0)
- **Hour 0 bitsandbytes/sm_120 smoke test BEFORE any training code** (RTX 5060 Ti is Blackwell arch)
- **CUDA 13.0 verified** (pyproject.toml cu130 is canonical; baseline cu128 is legacy artifact)
- **Reproducible seeds:** SEED=42 for primary runs, SEED=123 for ensemble diversity model

### Must NOT Have
- External API calls (OpenAI, Anthropic, etc.)
- Training on test data or any form of test label leakage
- Blind hyperparameter changes without validation measurement
- Deleting or overwriting the original baseline notebooks
- Training on dev samples with <4/5 annotator agreement without explicit justification
- Save paths pointing to `/content/` (Colab artifact -- use local project paths)

---

## Task Flow (Hour 0 + 4 Core Steps + 2 Stretch Goals)

### Hour 0: Environment Setup and Smoke Tests (Day 1, ~1 hour)

**Objective:** Verify that the entire toolchain works on the actual hardware BEFORE writing any training code. RTX 5060 Ti is Blackwell architecture (sm_120) -- bitsandbytes compatibility is NOT guaranteed.

**Checklist (execute in order, stop on first failure):**

1. **Verify CUDA version:**
   ```bash
   nvidia-smi   # Confirm CUDA 13.0 (NOT 13.1)
   python -c "import torch; print(torch.version.cuda)"  # Must match cu130
   ```
   - pyproject.toml specifies cu130 -- this is the canonical source
   - Baseline notebook used cu128 nightly -- this is a legacy artifact, ignore it

2. **Install missing dependencies:**
   ```bash
   uv add peft bitsandbytes
   ```

3. **Smoke-test bitsandbytes on sm_120 (CRITICAL):**
   ```python
   import bitsandbytes as bnb
   import torch
   # Test that 4-bit quantization works on Blackwell GPU
   linear = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16)
   linear = linear.cuda()
   x = torch.randn(1, 64, dtype=torch.bfloat16, device="cuda")
   out = linear(x)
   print(f"bitsandbytes 4bit OK on {torch.cuda.get_device_name()}")
   ```
   - **If this fails:** bitsandbytes does not support sm_120 yet
   - **Fallback:** Skip 4-bit quant entirely. Use fp16/bfloat16 training of 3B model only (fits in 16GB without quantization). 7B without quant does NOT fit -- 7B path is blocked.
   - **Fallback training config:** Qwen2.5-VL-3B-Instruct, bfloat16 (no quantization), LoRA r=16, batch_size=1, grad_accum=8

4. **Smoke-test model loading:**
   ```python
   from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
   from peft import get_peft_model, LoraConfig
   # Test 3B loads with quantization (or without if fallback)
   model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
       "Qwen/Qwen2.5-VL-3B-Instruct",
       quantization_config=bnb_config,  # or omit if fallback
       device_map="auto"
   )
   print(f"Peak VRAM after load: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
   ```

5. **Verify peft + bitsandbytes in pyproject.toml:**
   ```bash
   grep -E "peft|bitsandbytes" pyproject.toml
   ```

**Go/No-Go Decision:**
- bitsandbytes works on sm_120 --> proceed with full plan (4-bit quant, 7B path open)
- bitsandbytes fails on sm_120 --> fallback mode: fp16 3B only, skip Step 2 (7B), adjust all downstream steps accordingly

**Acceptance Criteria:**
- [ ] CUDA 13.0 confirmed via nvidia-smi and torch
- [ ] bitsandbytes 4-bit linear test passes on GPU (or fallback documented)
- [ ] 3B model loads successfully with quantization (or fp16 fallback)
- [ ] Peak VRAM after model load logged
- [ ] peft and bitsandbytes present in pyproject.toml

**Estimated time:** ~30-60 minutes (including dependency install and download)

---

### Step 1: Foundation -- Label Masking + Full Data + Fixed 3B (Day 1, ~6 hours)

**Objective:** Fix all 6 critical baseline issues and establish a strong 3B benchmark. This is the highest-ROI step in the entire plan.

**Changes to make:**

0. **(Moved to Hour 0)** -- Dependencies and environment setup are completed in Hour 0.

1. **Label masking in DataCollator (CRITICAL -- highest priority change):**
   ```python
   # In DataCollator.__call__:
   # After tokenizing the full conversation, mask all tokens EXCEPT the answer.
   #
   # Implementation (Qwen2.5-VL uses ChatML format):
   #
   #   1. Tokenize the full conversation (system + user + assistant response)
   #   2. Find the last occurrence of the assistant header token sequence:
   #      Look for: <|im_start|>assistant\n
   #      This is typically token IDs for "<|im_start|>" + "assistant" + "\n"
   #   3. The answer token is the FIRST content token AFTER this header sequence
   #      i.e., assistant_answer_start = position_after_header_newline
   #   4. Set labels[:, :assistant_answer_start] = -100
   #   5. Keep only the answer token (e.g., "a") in labels
   #   6. Optionally keep <|im_end|> in labels (minor effect)
   #
   # Common off-by-one error: The \n after "assistant" is part of the header,
   # NOT the answer. Make sure assistant_answer_start points to the token
   # AFTER the \n, not the \n itself.
   #
   # Verification: Print labels tensor for one sample. It should be:
   #   [-100, -100, ..., -100, answer_token_id, <optional im_end id>]
   #   with exactly 1-2 non-(-100) values at the end.
   ```

2. **Create `train_vqa.py`** (standalone Python script, not notebook -- faster iteration):
   - Load ALL 5,073 train samples (remove `train_df.sample(n=200)`)
   - Split: 4,566 train / 507 validation (90/10 stratified by answer AND question type)
   - Fix image paths to use absolute paths from project root (not `/content/...`)
   - Fix `extract_choice()` default: instead of returning `"a"`, return the answer with highest log-probability among {a,b,c,d}. This eliminates silent misparses from the start.

3. **Korean system prompt (SYSTEM_INSTRUCT only -- user prompt is already Korean):**
   ```python
   # Change ONLY the SYSTEM_INSTRUCT to Korean. The user prompt (question + choices)
   # is already in Korean from the dataset. Do NOT translate or modify user prompts.
   SYSTEM_INSTRUCT = "당신은 재활용 관련 이미지를 보고 질문에 답하는 전문가입니다. 보기 중 정확한 답을 a, b, c, d 중 하나의 소문자로만 답하세요."
   ```

4. **Increase IMAGE_SIZE:** 384 -> 504 (sweet spot for 3B + 16GB VRAM)
   - Set `min_pixels=504*504, max_pixels=504*504` in processor

5. **Training hyperparameters for 3B:**
   - **seed: 42** (primary seed for all reproducible runs)
   - epochs: 3
   - batch_size: 1, grad_accum: 8 (effective batch 8)
   - lr: 2e-5 (lower than baseline 1e-4 -- less aggressive)
   - warmup: 10% of total steps
   - LoRA r=16, alpha=32 (double the baseline)
   - weight_decay: 0.01
   - cosine scheduler (replace linear)
   - **save_strategy: per-epoch** (save checkpoint after each epoch to prevent total loss on crash; use `save_strategy="epoch"` in TrainingArguments or manual `model.save_pretrained()` after each epoch)
   - **bnb_4bit_compute_dtype: bfloat16** (baseline uses float16 -- bfloat16 is more numerically stable)

6. **Create `inference_vqa.py`** (standalone inference script):
   - Load saved LoRA adapter
   - Use log-probability extraction for answer selection (not text generation + parsing)
   - **Use batched inference** (batch_size=4-8 depending on VRAM headroom). Single-sample inference on 5,074 test samples takes 30-60 minutes. Batching reduces this to ~10-15 minutes.
   - Generate `submission.csv`
   - Save path: `./outputs/submission_3b.csv` (not `/content/...`)

7. **Validation must be stratified by question type** (counting vs material), not just by answer label.

**Acceptance Criteria:**
- [ ] Label masking verified: loss is computed on answer token(s) only (check by printing labels tensor -- should be mostly -100)
- [ ] Training completes on all 4,566 train samples for 3 epochs without OOM
- [ ] Checkpoint saved after each epoch (3 checkpoint dirs exist)
- [ ] Validation accuracy measured and printed after each epoch, stratified by question type
- [ ] First submission.csv generated and verified (correct format, 5074 rows, all answers in {a,b,c,d})
- [ ] Validation accuracy >= 55% (vs ~25% random baseline)
- [ ] `peft` and `bitsandbytes` present in pyproject.toml

**Go/No-Go to Step 2:** Proceed to Step 2 (7B) ONLY if val accuracy >= 50%. If < 50%, debug label masking and data pipeline before scaling up. A sub-50% result means something fundamental is broken.

**Estimated time:** ~2h coding + ~4.1h training (3 epochs x ~82min/epoch at 504px) = ~6h total

---

### Step 2: Scale Up -- Qwen2.5-VL-7B Fine-tuning (Day 2, ~8 hours)

**Objective:** Switch to the 7B model for a capability jump. This is the primary model.

**Changes to make:**

1. **Model swap:**
   ```python
   MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
   ```

2. **VRAM-optimized configuration for 7B:**
   - **seed: 42** (same primary seed as 3B for fair comparison)
   - 4-bit quantization (NF4, double quant, **bnb_4bit_compute_dtype=bfloat16**)
   - IMAGE_SIZE: 504 initially (plan target). **Resolution fallback chain if OOM:**
     - 504px OOMs --> reduce to 448px (still above baseline 384px)
     - 448px OOMs --> reduce to 384px (baseline level, still worth running 7B)
     - 384px OOMs --> fall back to 3B entirely (see fallback plan below)
     - NOTE: 448px and 384px are DECREASES from the 504px target, NOT increases from baseline
   - batch_size: 1, grad_accum: 8
   - gradient_checkpointing: True (already enabled)
   - LoRA r=16, alpha=32, targeting same modules
   - lr: 1e-5 (lower for larger model)
   - **epochs: 2** (not 3 -- 7B at 504px takes ~2.6h/epoch, 3 epochs = ~7.8h which is too long for Day 2)
   - save_strategy: per-epoch (save checkpoint after each epoch)

3. **Memory monitoring (not try/catch -- CUDA OOM is not catchable in Python):**
   - Log `torch.cuda.max_memory_allocated()` after first training step
   - If peak VRAM > 15GB after step 1, immediately stop and reduce IMAGE_SIZE to 448
   - Add `torch.cuda.empty_cache()` before training starts

4. **Fallback plan if 7B OOMs:**
   - First: reduce IMAGE_SIZE 504 --> 448 and retry
   - Second: reduce IMAGE_SIZE 448 --> 384 and retry
   - Last resort: fall back to 3B with IMAGE_SIZE=672 and more epochs (5)

**Acceptance Criteria:**
- [ ] 7B model loads successfully with 4-bit quant (peak VRAM < 14GB after load)
- [ ] First training step completes; peak VRAM logged and < 15.5GB
- [ ] Training completes for 2 epochs with per-epoch checkpoints saved
- [ ] Validation accuracy > Step 1 accuracy (expected: 60-70%)
- [ ] Second submission uploaded

**Go/No-Go to Step 3:** Proceed to Step 3 ONLY if 7B accuracy > 3B accuracy. If 7B accuracy <= 3B accuracy, stay with 3B as the primary model for Steps 3-4. The 7B model still serves as an ensemble partner regardless.

**Estimated time:** ~1h setup + ~5.2h training (2 epochs x ~156min/epoch at 504px) + ~1h inference + buffer = ~8h total

---

### Step 3: Data Expansion -- High-Confidence Dev Pseudo-Labels as A/B Test (Day 3, ~7 hours)

**Objective:** Test whether adding carefully filtered dev samples improves accuracy. Do NOT assume it helps -- run an A/B comparison.

**Changes to make:**

1. **Dev data filtering pipeline (threshold >=4/5 ONLY):**
   ```python
   # For each dev sample, compute majority vote across 5 annotators
   # Keep ONLY samples where >=4 out of 5 annotators agree
   # This yields 891 high-confidence samples (~95%+ label accuracy)
   #
   # Verified breakdown of the 891 samples (>=4/5 agreement subset):
   #   Question type: 702 counting / 189 non-counting (78.8% / 21.2%)
   #   Within non-counting: 77 material, 112 other
   #   Answer distribution: a=180(20.2%), b=187(21.0%), c=116(13.0%), d=408(45.8%)
   #   WARNING: MASSIVE d-skew in answer labels!
   #
   # DO NOT use >=3/5 threshold:
   #   - 74.3% of >=3 samples have only 3/5 agreement (42-75% accuracy)
   #   - Noisy labels at scale can degrade a fine-tuned model
   ```

2. **Dev distribution stratification (CRITICAL -- TWO skews exist simultaneously):**
   ```python
   # Skew 1 (question type): 78.8% counting vs train's 34.5% counting
   # Skew 2 (answer label): d=45.8% vs train's ~25% balanced
   #
   # BOTH skews must be addressed. Two stratification approaches:
   #
   # OPTION A: Answer-label stratification (recommended first pass)
   #   - Min class is c with 116 samples
   #   - Cap each answer label (a,b,c,d) to 116 samples
   #   - Result: 464 usable samples, answer-balanced
   #   - Question-type skew partially reduced but not eliminated
   #
   # OPTION B: Question-type + answer-label stratification (strictest)
   #   - After answer-label cap: further subsample to match train question-type ratio
   #   - Result: ~242 usable samples
   #   - Most balanced but fewest samples
   #
   # Decision: Use OPTION A (464 samples) for Version B.
   # Rationale: 464 >> 242, and the A/B test will reveal if residual
   # question-type skew causes harm.
   ```

3. **A/B Test (HARD REQUIREMENT -- do not skip):**
   ```python
   # Train TWO versions of the best model from Step 2:
   #
   # Version A: Train data only (4,566 samples), 2 epochs, seed=42
   #   - This is essentially a retrain of the Step 2 winner with same data
   #   - Serves as the control
   #
   # Version B: Train + stratified dev data (4,566 + 464 = 5,030 samples), 2 epochs, seed=42
   #   - Same hyperparameters as Version A
   #   - Only difference is the additional 464 stratified dev samples
   #
   # Compare on the 507 held-out validation set.
   # ONLY adopt Version B if accuracy >= Version A + 1pp
   # If Version B is worse or within 1pp: discard it, keep Version A
   ```

4. **Validation integrity:**
   - Validation set remains the 507 held-out train samples (never touched by dev data)
   - Report accuracy separately for counting and material question types
   - Report accuracy separately per answer label (a/b/c/d) to detect d-skew contamination

**Acceptance Criteria:**
- [ ] Dev filtering produces the correct count of >=4/5 agreement samples (verify against 891)
- [ ] Answer-label stratification produces 464 samples (116 per answer label)
- [ ] Version A trained and validated
- [ ] Version B trained and validated
- [ ] A/B comparison logged: Version A accuracy vs Version B accuracy
- [ ] Winner selected by >= 1pp improvement rule
- [ ] No label leakage between validation and training splits
- [ ] Third submission uploaded (winner of A/B test)

**Go/No-Go to Step 4:** Proceed to Step 4 regardless of A/B outcome. Ensemble helps even if dev data is dropped. Carry forward whichever version won the A/B test.

**Estimated time:** ~1h coding + ~3h training Version A + ~3h training Version B = ~7h total (Versions can overlap if checkpointing is used efficiently)

---

### Step 4: Inference Optimization + 2-Model Ensemble (Day 3-4, ~6 hours)

**Objective:** Maximize final accuracy through log-prob inference, a second diverse model, and ensemble.

**Changes to make:**

1. **Log-probability based answer selection (if not already done in Step 1):**
   ```python
   # Instead of generating text and parsing, compute logits for next token
   # Compare P(a), P(b), P(c), P(d) directly from the model's logit output
   # Pick argmax -- eliminates parsing errors entirely
   #
   # Implementation:
   #   1. Prepare input with the full prompt (system + user question + choices)
   #   2. Run model forward pass (no generation)
   #   3. Extract logits at the last input token position
   #   4. Get log-probs for token IDs corresponding to "a", "b", "c", "d"
   #   5. Answer = argmax over those 4 log-probs
   #
   # Note: Must handle tokenizer encoding of "a"/"b"/"c"/"d" carefully
   # (may map to different token IDs depending on context/spacing)
   ```

2. **Train second diverse model for ensemble:**
   - If Step 2 used 7B: train a 3B model with IMAGE_SIZE=672, lr=2e-5, **seed=123** (different from primary seed=42 for diversity), 3 epochs (~4.1h)
   - If Step 2 used 3B (fallback): train a second 3B with **seed=123**, IMAGE_SIZE=504, lr=1e-5
   - Both models should use the winning dataset from Step 3 A/B test

3. **2-model ensemble (exactly 2 models, not 3-5):**
   ```python
   # For each test sample:
   #   1. Get log-probability distributions from each model
   #   2. Weighted average: P_final = w1*P_model1 + w2*P_model2
   #   3. Weights = normalized validation accuracies of each model
   #   4. Final answer = argmax(P_final)
   #
   # Use batched inference (batch_size=4-8) for BOTH models.
   # Single-sample inference on 5,074 test samples = 30-60min PER MODEL.
   # Batching reduces to ~10-15min per model.
   #
   # Fallback (if log-prob extraction fails for one model):
   #   Majority vote, ties broken by higher-accuracy model
   ```

4. **Submission management:**
   - Track every submission: model, data, prompt, val accuracy, leaderboard score
   - Upload best single-model as backup
   - Upload ensemble as primary

**Acceptance Criteria:**
- [ ] Log-probability extraction verified (no parsing needed, 0% parse failures)
- [ ] Second model trained with different configuration than primary
- [ ] Ensemble prediction script produces correct CSV
- [ ] Ensemble accuracy >= best single model accuracy on validation set
- [ ] Both single-model and ensemble submissions uploaded
- [ ] All submissions tracked in a log file

**Estimated time:** ~1h log-prob coding + ~4h training second model + ~1h ensemble/inference = ~6h total

---

### STRETCH GOAL A: Prompt Engineering (Day 4, if ahead of schedule)

**Only attempt if Steps 1-4 complete with time remaining.**

**Prompt variants to A/B test on validation:**

1. **Question-type-aware prompt:**
   ```python
   if any(kw in question for kw in ['몇 개', '몇개', '개수', '몇 종류']):
       system = "당신은 이미지 속 재활용 물품의 개수를 정확히 세는 전문가입니다. ..."
   else:
       system = "당신은 재활용 물품의 재질과 분류를 판별하는 전문가입니다. ..."
   ```

2. **Choice-order augmentation during training:**
   ```python
   # Randomly shuffle (a,b,c,d) order during training
   # Forces model to read choices, not memorize position bias
   ```

**Acceptance Criteria:**
- [ ] At least 1 prompt variant tested with measured accuracy delta
- [ ] Only adopt if validation accuracy improves by >= 1pp

---

### STRETCH GOAL B: Test-Time Augmentation (Day 4, if ahead of schedule)

**Only attempt if Steps 1-4 complete AND Stretch A tested.**

```python
# For each test image, run inference on:
#   - Original image
#   - Horizontally flipped image
# Average log-probabilities across augmentations
# Pick argmax of averaged probs
```

**Acceptance Criteria:**
- [ ] TTA implemented and tested on validation set
- [ ] Only submit if validation accuracy improves over non-TTA ensemble

---

## Detailed Timeline (Revised with Measured Estimates)

| Day | Morning (4h) | Afternoon (4h) | Evening (4h) |
|-----|-------------|-----------------|---------------|
| **Day 1** (Apr 2) | **Hour 0:** Env setup, CUDA 13.0 verify, bitsandbytes sm_120 smoke test, go/no-go. Then Step 1: implement label masking, build train_vqa.py + inference_vqa.py | Step 1: Start 3B training (~4.1h, runs into evening) | Step 1: 3B training completes, validate, first submission. Go/no-go check (val >= 50%) |
| **Day 2** (Apr 3) | Step 2: 7B setup, verify VRAM, start training | Step 2: 7B training continues (~5.2h total) | Step 2: 7B validation + submission. Go/no-go check (7B > 3B?). Step 3: Build dev filtering + stratification pipeline |
| **Day 3** (Apr 4) | Step 3: Train Version A (~3h) | Step 3: Train Version B (~3h), A/B comparison, submit winner | Step 4: Log-prob inference coding + start second model training (seed=123) |
| **Day 4** (Apr 5) | Step 4: Second model training completes, ensemble inference + submit | Stretch A: prompt experiments. Stretch B: TTA if ahead | Final submissions, safety backup uploads |

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **bitsandbytes fails on sm_120 (Blackwell)** | **MEDIUM** | **CRITICAL** | **Hour 0 smoke test catches this immediately. Fallback: fp16 training of 3B only (no quantization). 7B path blocked without 4-bit quant.** |
| 7B model OOMs on 16GB | MEDIUM | HIGH | Resolution fallback chain: 504 --> 448 --> 384 --> fall back to 3B@672. Each step is a decrease from the plan's 504px target, not an increase from baseline. |
| Training crash mid-epoch (9h wasted) | MEDIUM | CRITICAL | **Per-epoch checkpoint saving** -- maximum loss is 1 epoch |
| Dev pseudo-labels hurt accuracy | LOW-MEDIUM | MEDIUM | **A/B test is mandatory** (Step 3). Version A (train only) vs Version B (train+dev). Only adopt if >= 1pp improvement. |
| Overfitting on small train set | MEDIUM | MEDIUM | 2-3 epochs (not 5), early stopping, LoRA regularization, monitor val loss per epoch |
| Submission format error | LOW | HIGH | Automated format checker before every submission |
| Training instability (NaN loss) | LOW | MEDIUM | Use bfloat16 (not float16) for compute dtype; gradient clipping at 1.0 |
| Running out of time | MEDIUM | HIGH | Safe submission uploaded by end of Day 2; every subsequent change is incremental |
| extract_choice silent failures | HIGH (in baseline) | MEDIUM | Log-prob extraction from Step 1 eliminates parsing entirely |
| Slow inference blocks iteration | MEDIUM | MEDIUM | **Batched inference** (batch_size=4-8). Single-sample = 30-60min; batched = 10-15min for 5,074 samples. |

## Expected Accuracy Progression (Recalibrated)

| Stage | Expected Val Accuracy | Delta | Notes |
|-------|----------------------|-------|-------|
| Random baseline | 25% | -- | |
| Current baseline (200 samples, 1 epoch, no label mask) | ~30-35% | -- | Most loss signal wasted on non-answer tokens |
| Step 1: Label masking + full data + fixed hyperparams (3B) | 55-65% | +25-30pp | Label masking alone: +5-15pp. Full data: +10-15pp |
| Step 2: Upgrade to 7B | 60-70% | +3-7pp | Realistic 7B uplift is 3-7pp, not 10pp |
| Step 3: A/B test with high-conf dev pseudo-labels (>=4/5 only) | 62-72% | +0-3pp | Only 464 answer-stratified samples -- modest impact. A/B test determines if adopted. |
| Step 4: Ensemble (2 models) + log-prob inference | 65-75% | +1-3pp | 2-model ensemble ceiling is lower than 5-model |
| Stretch A: Prompt engineering | +0-2pp | conditional | Only if measurably helps on validation |
| Stretch B: TTA | +0-1pp | conditional | Diminishing returns |
| **Realistic final range** | **65-75%** | | Honest range based on measured benchmarks |

## Key Hyperparameter Reference

| Parameter | 3B Model | 7B Model |
|-----------|----------|----------|
| MODEL_ID | Qwen/Qwen2.5-VL-3B-Instruct | Qwen/Qwen2.5-VL-7B-Instruct |
| **Seed** | **42 (primary), 123 (ensemble diversity)** | **42** |
| Quantization | 4-bit NF4, double quant | 4-bit NF4, double quant |
| bnb_4bit_compute_dtype | **bfloat16** | **bfloat16** |
| IMAGE_SIZE | 504 (ensemble variant: 672) | 504 (fallback chain: 448 --> 384 if OOM) |
| LoRA r | 16 | 16 |
| LoRA alpha | 32 | 32 |
| LoRA dropout | 0.05 | 0.05 |
| LoRA targets | q,k,v,o,gate,up,down _proj | q,k,v,o,gate,up,down _proj |
| Learning rate | 2e-5 | 1e-5 |
| Scheduler | cosine with warmup | cosine with warmup |
| Warmup ratio | 0.10 | 0.10 |
| Epochs | 3 | **2** |
| Batch size | 1 | 1 |
| Grad accumulation | 8 | 8 |
| Weight decay | 0.01 | 0.01 |
| Max grad norm | 1.0 | 1.0 |
| Save strategy | per-epoch | per-epoch |
| Label masking | Yes (-100 for non-answer tokens) | Yes (-100 for non-answer tokens) |
| MAX_NEW_TOKENS | N/A (log-prob mode) | N/A (log-prob mode) |

## ADR (Architecture Decision Record)

**Decision:** Use Qwen2.5-VL-7B-Instruct as primary model with 4-bit quantization, LoRA fine-tuning, and label masking. Ensemble with a single 3B variant (seed=123) for diversity. Maximum 2 models in ensemble. Dev pseudo-labels adopted ONLY if A/B test shows >= 1pp improvement.

**Drivers:**
1. 16GB VRAM limit eliminates models above 7B
2. Single-GPU timeline (~48h) limits total training to ~3-4 full training runs
3. Korean language VQA requires strong multilingual vision-language model
4. Label masking is the single highest-impact fix, applicable to any model choice
5. RTX 5060 Ti (Blackwell/sm_120) compatibility with bitsandbytes is unverified -- Hour 0 smoke test is a hard gate

**Alternatives considered:**
- InternVL2.5-4B/InternVL3-2B: Strong benchmarks but requires ~1 day of code restructuring. On a 4-day timeline with a single GPU, this cost is prohibitive. Only viable as a stretch diversity candidate.
- LLaVA-1.6-7B: Weaker Korean support than Qwen2.5-VL
- Qwen2.5-VL-3B only: Lower capability ceiling, insufficient for top-1. However, becomes primary if bitsandbytes fails on sm_120.
- 3-5 model ensemble: Sequential training on single GPU requires 20-40h just for training. Reduced to 2 models to stay within time budget.
- Dev pseudo-labels at >=3/5 threshold: 74.3% of qualifying samples have only 42-75% label accuracy. Raised to >=4/5 (891 samples at ~95%+ accuracy, 464 after answer-label stratification).
- Unconditional dev data inclusion: Rejected. A/B test required because dev has massive d-skew (45.8%) and question-type skew (78.8% counting). Contamination risk is real.

**Why chosen:** Qwen2.5-VL-7B is the largest model that fits in 16GB with 4-bit quant, has excellent Korean language support, and requires minimal code changes from the existing 3B baseline. Label masking + full data + 7B is the highest-ROI path within the time constraint. The 2-model ensemble (7B + 3B with seed=123) adds diversity without exceeding the GPU time budget. The A/B test for dev data ensures we never degrade accuracy by adding noisy/skewed pseudo-labels.

**Consequences:**
- 7B training is ~2x slower than 3B, limiting to 2 epochs per run
- Only 2 models in ensemble (lower diversity ceiling than 5-model)
- Realistic accuracy ceiling of 65-75% (not 76-85% as originally projected)
- Step 3 now takes ~7h instead of ~4h due to mandatory A/B test
- Stretch goals (prompt engineering, TTA) are genuinely stretch -- may not be reached
- If bitsandbytes fails on sm_120, 7B path is completely blocked (fp16 3B only)

**Follow-ups:**
- Hour 0: bitsandbytes sm_120 smoke test determines whether 7B path is viable
- Monitor Qwen2.5-VL-7B VRAM usage on first training step; fall back chain: 504 --> 448 --> 384 --> 3B
- After Day 2 leaderboard feedback, decide whether to invest time in prompt engineering or more training epochs
- If 7B performs worse than expected, pivot to 3B with more epochs and larger IMAGE_SIZE
- Track d-skew contamination: monitor per-answer-label accuracy after Step 3 A/B test

---

## Changes from v1 (Consensus Feedback Log -- iteration 1)

| # | Issue | Severity | Resolution |
|---|-------|----------|------------|
| 1 | Label masking missing | CRITICAL | Added as Change #0 in Step 1. Highest priority. |
| 2 | Timeline infeasible by 2-3x | CRITICAL | Rebuilt with measured benchmarks. 7B reduced to 2 epochs. |
| 3 | Dev threshold too low (>=3/5) | CRITICAL | Raised to >=4/5 (891 samples). Stratified by question type. |
| 4 | Dev distribution skew unaddressed | MAJOR | Added stratified subsampling to match train ratio. |
| 5 | Ensemble too large (3-5 models) | MAJOR | Reduced to exactly 2 models. |
| 6 | Missing peft/bitsandbytes deps | MAJOR | Added `uv add` as Step 1 prerequisite. |
| 7 | No checkpoint strategy | MAJOR | Added per-epoch checkpoint saving. |
| 8 | extract_choice defaults to "a" | MINOR | Fixed in Step 1 with log-prob extraction. |
| 9 | Validation not stratified by question type | MINOR | Added stratified validation. |
| 10 | Colab save path /content/ | MINOR | Fixed to local project path ./outputs/. |
| 11 | float16 for bnb compute dtype | MINOR | Changed to bfloat16. |
| 12 | CUDA OOM "try/catch" claim | MINOR | Removed. CUDA OOM is not catchable. Replaced with VRAM monitoring + preemptive fallback. |
| 13 | Accuracy trajectory overestimated | MINOR | Recalibrated to 65-75% final range. |
| 14 | Plan had 6 equal steps | STRUCTURAL | Collapsed to 4 core + 2 stretch. Critic's insight adopted: depth > breadth. |
| 15 | Corrected dev sample counts | MINOR | agreement=5: 0 (not 1), >=4: 891, >=3: 3,508 |

## Changes from v2 (Consensus Feedback Log -- iteration 2, FINAL)

| # | Issue | Severity | Resolution |
|---|-------|----------|------------|
| 16 | Hour 0 env setup missing; bitsandbytes/peft not installed; RTX 5060 Ti (sm_120) compatibility unverified | CRITICAL | Added Hour 0 with concrete smoke tests. If bitsandbytes fails on sm_120, fallback = fp16 3B only. |
| 17 | CUDA version stated as 13.1 (wrong) | CRITICAL | Corrected to CUDA 13.0 everywhere. pyproject.toml cu130 is canonical. Baseline cu128 is legacy artifact. |
| 18 | Dev stratification numbers were inaccurate | MAJOR | Corrected: 702 counting / 189 non-counting (78.8%/21.2%). Answer distribution: a=180, b=187, c=116, d=408. TWO simultaneous skews documented. |
| 19 | Step 3 assumed dev data helps without testing | MAJOR | Converted to mandatory A/B test. Version A (train only) vs Version B (train+dev). >= 1pp improvement required to adopt. Adds ~3h. |
| 20 | Label masking implementation too vague for executor | MAJOR | Added concrete ChatML guidance: find `<\|im_start\|>assistant\n` token sequence, set labels[:, :assistant_answer_start] = -100, warned about \n off-by-one error. |
| 21 | 448px fallback misleadingly described | MAJOR | Clarified: 448px is a DECREASE from plan's 504px target, NOT an increase from baseline 384px. Added full fallback chain: 504 --> 448 --> 384 --> 3B. |
| 22 | No go/no-go thresholds between steps | MINOR | Added: Step 1-->2: val >= 50%. Step 2-->3: 7B > 3B. Step 3-->4: proceed regardless. |
| 23 | Checkpoint saving mentioned but not concrete | MINOR | Specified `save_strategy="epoch"` or manual `model.save_pretrained()` per epoch. |
| 24 | Inference batching not mentioned | MINOR | Added: batch_size=4-8, single-sample = 30-60min, batched = 10-15min for 5,074 test samples. |
| 25 | Korean prompt ambiguity | MINOR | Clarified: change SYSTEM_INSTRUCT to Korean only. User prompt already contains Korean from dataset. |
| 26 | No seed specification | MINOR | Added: SEED=42 for primary runs, SEED=123 for ensemble diversity model. |
