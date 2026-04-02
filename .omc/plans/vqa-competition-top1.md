# VQA Competition TOP-1 Delta Plan (v5)

**Created:** 2026-04-02
**Revised:** 2026-04-02 (v5 -- DELTA plan on top of existing sophisticated notebook)
**Deadline:** 2026-04-06 (4 days remaining)
**Objective:** Push from current strong baseline to TOP-1 accuracy on SSAFY 15th AI Challenge recycling image VQA

---

## CRITICAL: What Already Exists

The notebook `ssafy_vqa_qwen25vl_h100_topscore_notebook_local.ipynb` is a sophisticated, production-quality implementation. **This delta plan does NOT rebuild anything.** It modifies the existing notebook.

### Already Implemented (DO NOT REWRITE)
| Feature | Status | Current Config |
|---------|--------|----------------|
| Qwen2.5-VL-7B with 4-bit quant + LoRA | DONE | model_id = "Qwen/Qwen2.5-VL-7B-Instruct" |
| Label masking (prefix tokens = -100) | DONE | prefix_lens from enc_prefix, non-answer masked |
| Full training data (all 5,073 samples) | DONE | official_train_core with sample_weight=1.0 |
| Dev pseudo-labels with confidence filtering | DONE | shared >=0.80 (4/5), count >=0.60 (3/5) |
| TrashNet synthetic Korean VQA | DONE | max 2,500 samples, weight=0.55 |
| TACO synthetic Korean VQA | DONE | max 1,800 samples, shared=0.45, count=0.75 |
| Question-type-aware dual system prompts | DONE | SYSTEM_PROMPT_GENERAL + SYSTEM_PROMPT_COUNT |
| Choice shuffling during training | DONE | shuffle_options=True |
| Multi-adapter training | DONE | shared adapter (1 epoch) + count adapter (2 epochs) |
| Log-probability answer extraction | DONE | permutation-based log-prob ensemble |
| OCR integration (EasyOCR + rapidfuzz) | DONE | OCR channel in ensemble for text-type questions |
| Per-question-type weighted ensemble | DONE | 7 question types x 4 channels (shared/count/text/ocr) |
| Permutation-based inference | DONE | n_perm_shared=2, n_perm_count=3 |
| Sample weighting | DONE | per-source weights (official=1.0, dev/trashnet/taco vary) |
| Dynamic resolution | DONE | train: 448-1280px, infer: 512-1536px |
| JPEG augmentation | DONE | prob=0.35, quality 55-95 |
| Stratified validation split | DONE | stratified by qtype + answer |
| Merger module unfreezing | DONE | unfreeze_full_module_keywords=("merger",) |
| Gradient checkpointing | DONE | use_reentrant=False |
| Linear scheduler with warmup | DONE | warmup_ratio=0.03 |
| LoRA r=32, alpha=64, dropout=0.05 | DONE | targets: q/k/v/o/gate/up/down_proj |

### Current Hyperparameters (Baseline for Deltas)
```
model_id        = "Qwen/Qwen2.5-VL-7B-Instruct"
lora_r          = 32,  lora_alpha = 64,  lora_dropout = 0.05
shared_epochs   = 1,   count_epochs = 2
lr_shared       = 8e-5, lr_count = 1e-4
warmup_ratio    = 0.03, weight_decay = 0.01, max_grad_norm = 1.0
train_min_pixels = 448*448,  train_max_pixels = 1280*1280
infer_min_pixels = 512*512,  infer_max_pixels = 1536*1536
per_device_batch = 1,  grad_accum = 8
dev_shared_min_conf = 0.80 (4/5),  dev_count_min_conf = 0.60 (3/5)
trashnet_max = 2500, taco_max = 1800
n_perm_shared = 2,  n_perm_count = 3
scheduler = linear with warmup
```

---

## Delta Plan: 4 Targeted Improvements

### ROI Assessment (Honest Expected Gains)

| Delta | Expected Gain | Confidence | Risk | Time Cost |
|-------|--------------|------------|------|-----------|
| Delta 1: Qwen3-VL-8B upgrade | +3-8pp | MEDIUM | VRAM overflow, Korean degradation | ~2h setup + retrain |
| Delta 2: KVQA + TOD external data | +1-3pp | MEDIUM-LOW | Data noise, format mismatch | ~3h pipeline + retrain |
| Delta 3: Hyperparameter tuning | +1-3pp | MEDIUM | Diminishing returns on strong baseline | ~2h per A/B run |
| Delta 4: Technique improvements | +0-2pp | LOW | Minimal downside | ~1-2h |
| **Combined realistic gain** | **+3-10pp** | | | |

**Honesty note:** The existing notebook is already highly optimized. Each delta has diminishing returns. The model upgrade (Delta 1) is the highest-ROI change. External data (Delta 2) helps only if the data quality is high enough. Hyperparameter tuning (Delta 3) and technique tweaks (Delta 4) are marginal.

---

### Delta 1: Model Upgrade -- Qwen3-VL-8B-Instruct (HIGHEST PRIORITY)

**Rationale:** Qwen3-VL-8B benchmarks substantially exceed Qwen2.5-VL-7B across the board.

| Benchmark | Qwen2.5-VL-7B (current) | Qwen3-VL-8B (target) | Gap |
|-----------|------------------------|-----------------------|-----|
| MMMU | 58.6 | 69.6 | +11.0pp |
| DocVQA | 95.7 | 96.1 | +0.4pp |
| OCRBench | 864 | 896 | +32 |
| MathVista | 68.2 | 77.2 | +9.0pp |
| MMBench | 82.6 | 85.0 | +2.4pp |
| Korean VQA (KOFFVQA) | 67.7 | N/A (untested) | Unknown |

**Hour 0: VRAM Smoke Test (HARD GATE)**

1. Load Qwen3-VL-8B with identical 4-bit quant config:
   ```python
   from transformers import AutoModelForVision2Seq, AutoProcessor
   # NOTE: Check actual model class name on HuggingFace model card.
   # May need Qwen3VLForConditionalGeneration if AutoModel doesn't resolve.
   model = AutoModelForVision2Seq.from_pretrained(
       "Qwen/Qwen3-VL-8B-Instruct",
       quantization_config=bnb_config,  # same NF4 config as current
       device_map="auto"
   )
   print(f"Peak VRAM after load: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
   ```

2. Go/No-Go thresholds:
   - Peak VRAM <= 12GB after load: **GO** at current resolution (448-1280 train)
   - Peak VRAM 12-14GB: **GO** with reduced train_max_pixels (896*896, already in the VRAM-constrained fallback)
   - Peak VRAM > 14GB: **NO-GO**. Stay with Qwen2.5-VL-7B. Skip Delta 1 entirely.

**Code Changes Required (MINIMAL -- same Qwen family):**

1. **CFG.model_id:** Change `"Qwen/Qwen2.5-VL-7B-Instruct"` to `"Qwen/Qwen3-VL-8B-Instruct"`

2. **Model class:** The notebook uses `Qwen2_5_VLForConditionalGeneration` (or AutoModel). Qwen3-VL may need:
   - `Qwen3VLForConditionalGeneration` (check transformers version)
   - Or `AutoModelForVision2Seq` (safest -- auto-resolves the correct class)
   - Verify: `from transformers import AutoModelForVision2Seq; print(type(model))`

3. **mm_token_type_ids:** Qwen3-VL processor generates this tensor. The existing collator must pass it through. Check if `processor(...)` output contains `mm_token_type_ids` and if so, ensure it is included in the batch dict sent to `model.forward()`. If the collator drops unknown keys, add it explicitly.

4. **Chat template verification:** Qwen3-VL may use a different chat template than ChatML. The label masking logic finds the prefix boundary via `enc_prefix`. Verify that:
   - `processor.apply_chat_template(...)` works with the same message format
   - The prefix/answer boundary is still correctly computed
   - Print one sample's labels tensor to confirm mostly -100 with 1-2 answer tokens at the end

5. **VRAM-constrained config (if 12-14GB):**
   - The notebook already has a fallback: `CFG.train_max_pixels = 896 * 896` (line 253)
   - May also need to reduce `infer_max_pixels` from 1536*1536 to 1280*1280

**Validation Gate (Korean Performance Check):**
- After training shared adapter (1 epoch), compare validation accuracy to the Qwen2.5-VL-7B baseline
- If Qwen3-VL-8B validation accuracy < Qwen2.5-VL-7B accuracy - 2pp: **FALL BACK to Qwen2.5-VL-7B**
- This catches the risk of Qwen3-VL having weaker Korean capabilities despite stronger general benchmarks

**Acceptance Criteria:**
- [ ] Qwen3-VL-8B loads with 4-bit quant; peak VRAM logged and within threshold
- [ ] `mm_token_type_ids` correctly passed through collator (or confirmed absent)
- [ ] Chat template produces correct label masking (labels tensor mostly -100, 1-2 answer tokens)
- [ ] Shared adapter training completes 1 epoch without OOM
- [ ] Validation accuracy compared to Qwen2.5-VL-7B baseline; go/no-go decision documented
- [ ] If NO-GO: reverted to Qwen2.5-VL-7B, no time wasted

**Estimated time:** ~1h smoke test + ~1h code changes + 1 epoch training for validation = ~3-4h total (including the epoch training time based on current ~2.5 s/step estimate for 8B)

**Fallback:** Stay with Qwen2.5-VL-7B. The existing notebook is already strong. Do NOT waste more than 4 hours on Delta 1 if issues arise.

---

### Delta 2: Additional External Data -- KVQA + TOD (MEDIUM PRIORITY)

**Rationale:** The notebook already uses TrashNet (2,500) and TACO (1,800). Adding KVQA and TOD provides additional Korean-language recycling-domain training data.

**Downloads (MIT license, instant access):**
```bash
# KVQA (Korean VQA -- 100K pairs)
git clone https://github.com/sktbrain/KVQA.git ./data/external/kvqa/

# TOD (Trash Object Detection -- 5K images, 33K annotations, Korean MoE standards)
git clone https://github.com/jms0923/tod.git ./data/external/tod/
```

**Step 2a: KVQA Integration**

1. Inspect KVQA format after download (question types, answer format, image paths)
2. Filter for competition-relevant subsets:
   - Number/counting questions (~9,300 claimed) -- helps with 34.5% counting questions
   - Object identification -- helps with material recognition
3. Convert to 4-choice multiple-choice format matching the notebook's data schema:
   - Must produce: `id, path, question, a, b, c, d, answer, qtype, source, sample_weight`
   - Generate plausible Korean distractors for open-ended questions
4. Quality filter: Only keep samples where the answer maps cleanly to recycling/counting categories
5. Target: 2,000-4,000 high-quality filtered samples
6. Set `sample_weight` = 0.5-0.7 (lower than official data, similar to existing trashnet_weight=0.55)

**Step 2b: TOD Synthetic VQA Generation**

1. Inspect TOD annotation format (10 recycling classes from Korean Ministry of Environment)
2. Generate synthetic Korean VQA pairs using the SAME template patterns as the existing `build_trashnet_synthetic()` and `build_taco_synthetic()` functions:
   - Material questions: "이 물체의 재질은 무엇인가요?"
   - Counting questions (if multiple objects): "이미지에 재활용 가능한 물품이 몇 개 있나요?"
   - Classification questions: "이 물품은 어떤 분리수거 카테고리에 해당하나요?"
3. Create `build_tod_synthetic()` function following the exact pattern of existing `build_trashnet_synthetic()` and `build_taco_synthetic()`
4. Target: 1,500-2,500 synthetic VQA samples
5. Set `sample_weight` = 0.45-0.55

**Step 2c: Integration into Existing Data Pipeline**

1. Add to CFG:
   ```python
   use_kvqa: bool = True
   use_tod: bool = True
   kvqa_max_samples: int = 3000
   tod_max_samples: int = 2000
   kvqa_weight: float = 0.55
   tod_weight: float = 0.50
   ```
2. Add KVQA/TOD data to `shared_parts` and `count_parts` lists (same pattern as trashnet/taco)
3. Validation set stays as held-out competition train samples ONLY

**A/B Test (Required):**
- Version A: Current notebook (official + dev + trashnet + taco) -- CONTROL
- Version B: Current notebook + KVQA + TOD -- TEST
- Compare on validation set. Adopt Version B ONLY if accuracy >= Version A + 0.5pp
- If Version B is worse: discard KVQA/TOD, external data adds noise for this domain

**Acceptance Criteria:**
- [ ] KVQA downloaded and format inspected; usable sample count documented
- [ ] TOD downloaded and format inspected; usable sample count documented
- [ ] `build_kvqa_filtered()` function created (follows existing pattern)
- [ ] `build_tod_synthetic()` function created (follows existing pattern)
- [ ] New data integrated into shared_parts/count_parts pipeline
- [ ] A/B test run: Version A accuracy vs Version B accuracy logged
- [ ] Decision documented with accuracy numbers

**Estimated time:** ~2h data pipeline coding + ~1h format inspection + retrain time for A/B test

**Fallback:** If data format is unexpectedly complex or A/B test shows degradation, drop KVQA/TOD entirely. The existing TrashNet + TACO pipeline is already good.

---

### Delta 3: Hyperparameter Tuning (LOW-MEDIUM PRIORITY)

**Rationale:** The current config is reasonable but conservative in some areas. Small tuning may yield 1-3pp. Only attempt AFTER Delta 1 (model upgrade) is resolved, because optimal hyperparameters depend on the model.

**Candidates for A/B testing (one at a time, not all at once):**

1. **More training epochs (HIGHEST ROI within this delta)**
   - Current: shared_epochs=1, count_epochs=2
   - Test: shared_epochs=2, count_epochs=3
   - Rationale: With 5,073 samples and LoRA r=32, 1 epoch for shared may underfit. The notebook already has per-epoch checkpointing logic, so overfitting is easy to detect.
   - Risk: Overfitting if data is too small. Monitor val loss per epoch.
   - Expected gain: +1-2pp if underfitting was the bottleneck

2. **Cosine scheduler instead of linear**
   - Current: `get_linear_schedule_with_warmup`
   - Test: `get_cosine_schedule_with_warmup` (same transformers import)
   - Rationale: Cosine decay typically gives slightly better fine-tuning results than linear. One-line change.
   - Risk: Negligible
   - Expected gain: +0-1pp

3. **Higher inference resolution**
   - Current: infer_min_pixels=512*512, infer_max_pixels=1536*1536
   - Test: infer_min_pixels=672*672, infer_max_pixels=1792*1792 (if VRAM allows at inference time -- no gradients needed)
   - Rationale: Inference VRAM is much lower than training. Larger images at inference capture more visual detail.
   - Risk: Slower inference. Check VRAM before committing.
   - Expected gain: +0-1pp

4. **More permutations at inference**
   - Current: n_perm_shared=2, n_perm_count=3
   - Test: n_perm_shared=3, n_perm_count=5
   - Rationale: More permutations reduce position bias in choice selection. Diminishing returns beyond ~5.
   - Risk: Linear increase in inference time (3/2 = 1.5x for shared, 5/3 = 1.67x for count)
   - Expected gain: +0-1pp

**Protocol:** Test each change independently against the current best model. Keep a change ONLY if validation accuracy improves by >= 0.5pp. Do NOT stack untested changes.

**Acceptance Criteria:**
- [ ] Each tested hyperparameter change has a logged A/B result (accuracy delta)
- [ ] Only adopted changes show >= 0.5pp improvement
- [ ] No blind stacking of untested changes

**Estimated time:** ~2h per A/B test (mostly training time). Budget for 2-3 tests max.

---

### Delta 4: Technique Improvements (LOWEST PRIORITY -- Stretch)

**Only attempt if Deltas 1-3 are complete and time remains.**

1. **Chain-of-thought for counting questions**
   - Modify SYSTEM_PROMPT_COUNT to encourage step-by-step counting before giving the answer
   - Example: "먼저 이미지 속 재활용 물품을 하나씩 세어주세요. 그 다음 보기 중 정확한 답을 소문자 하나로 답하세요."
   - Risk: May increase output token length, complicating log-prob extraction. Label masking must be adapted.
   - Only test if counting accuracy is notably lower than material accuracy.

2. **Ensemble weight optimization**
   - Current: Fixed weights per question type (7 types x 4 channels)
   - Test: Optimize weights on validation set via grid search or scipy.optimize
   - Rationale: Hand-tuned weights may not be optimal
   - Risk: Overfitting to validation set if too many parameters

3. **Test-Time Augmentation (TTA)**
   - Run inference on original + horizontally flipped image
   - Average log-probabilities across both
   - Risk: 2x inference time. Horizontal flip may confuse text/OCR questions.
   - Only apply TTA to non-OCR, non-text question types.

**Acceptance Criteria:**
- [ ] Each technique tested independently with logged accuracy delta
- [ ] Only adopted if >= 0.5pp improvement

---

## Decision Tree Summary

```
Hour 0: VRAM smoke test for Qwen3-VL-8B
  |
  +-- VRAM <= 12GB --> Delta 1: Full 8B upgrade at current resolution
  |     |
  |     +-- Korean val accuracy >= 7B - 2pp --> KEEP 8B, proceed to Delta 2
  |     +-- Korean val accuracy < 7B - 2pp --> REVERT to 7B, skip Delta 1
  |
  +-- VRAM 12-14GB --> Delta 1: 8B with reduced train_max_pixels=896*896
  |     |
  |     +-- Same Korean validation gate as above
  |
  +-- VRAM > 14GB --> SKIP Delta 1 entirely, stay with 7B
  
Delta 2: KVQA + TOD external data (parallel with Delta 1 data download)
  |
  +-- A/B test: +external vs no-external
  +-- Adopt only if >= 0.5pp improvement

Delta 3: Hyperparameter tuning (after Delta 1 resolved)
  |
  +-- Test epochs, scheduler, resolution, permutations one at a time
  +-- Adopt each only if >= 0.5pp improvement

Delta 4: Stretch techniques (only if time remains)
```

---

## Timeline (4 Days)

| Day | Block | Activity |
|-----|-------|----------|
| **Day 1** | Morning | Hour 0: Qwen3-VL-8B VRAM smoke test. In parallel: download KVQA + TOD. |
| | Afternoon | Delta 1: Code changes for 8B (model class, mm_token_type_ids, chat template). Start shared adapter training. |
| | Evening | Delta 1: Validate 8B Korean accuracy. Go/no-go decision. Begin Delta 2 data pipeline coding. |
| **Day 2** | Morning | Delta 2: Complete KVQA filter + TOD synthetic generation. Integrate into pipeline. |
| | Afternoon | Delta 2: A/B test (train with vs without KVQA+TOD). |
| | Evening | Submit best model so far (SAFE SUBMISSION). Evaluate Delta 2 results. |
| **Day 3** | Morning | Delta 3: Test epoch count increase (shared_epochs=2, count_epochs=3). |
| | Afternoon | Delta 3: Test scheduler change (cosine) and/or inference resolution. |
| | Evening | Submit improved model. Evaluate cumulative gains. |
| **Day 4** | Morning | Delta 3: Any remaining A/B tests (permutations, etc.). |
| | Afternoon | Delta 4: Stretch techniques if time allows. |
| | Evening | Final submission. Safety backup. |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Qwen3-VL-8B OOMs on 16GB | MEDIUM-HIGH | HIGH | Hour 0 VRAM test with clear thresholds. Fallback: stay with 7B (already strong). |
| Qwen3-VL-8B poor Korean performance | LOW-MEDIUM | HIGH | Validation gate after 1 epoch. Revert to 7B if accuracy drops > 2pp. |
| Qwen3-VL-8B model class not in transformers | LOW | MEDIUM | Use AutoModelForVision2Seq. Check transformers version >= 4.51.0. |
| mm_token_type_ids breaks collator | LOW | MEDIUM | Check processor output keys. Add to collator if present. |
| KVQA format incompatible | LOW-MEDIUM | LOW | Inspect before coding pipeline. Drop if too complex. |
| TOD format incompatible | LOW-MEDIUM | LOW | Inspect before coding pipeline. Drop if too complex. |
| External data degrades performance | MEDIUM | LOW | A/B test is mandatory. Drop external data if no improvement. |
| Hyperparameter changes cause regression | LOW | LOW | A/B test each change independently. Revert on regression. |
| Time wasted on marginal improvements | MEDIUM | MEDIUM | Strict 4h cap on Delta 1 (including fallback). Strict A/B gates (>= 0.5pp) for all other deltas. |

---

## ADR (Architecture Decision Record)

**Decision:** Apply 4 targeted deltas to an already-sophisticated notebook: (1) model upgrade to Qwen3-VL-8B, (2) additional external data KVQA+TOD, (3) hyperparameter A/B tests, (4) stretch technique improvements. Each delta is independently testable and revertible.

**Drivers:**
1. Existing notebook already implements all major techniques (label masking, multi-adapter, log-prob, OCR, ensemble, etc.)
2. Qwen3-VL-8B is +11pp MMMU over current Qwen2.5-VL-7B -- largest single lever remaining
3. KVQA (100K Korean VQA) and TOD (5K recycling) are freely available and domain-relevant
4. 16GB VRAM constraint makes 8B upgrade risky but feasible with 4-bit quant
5. 4-day deadline requires strict ROI prioritization -- no time for speculative changes

**Alternatives considered:**
- Complete rewrite of the notebook: Rejected. The existing implementation is sophisticated and well-tested. A rewrite risks regression with no guaranteed improvement.
- InternVL or other model family: Rejected. Different processor API requires ~1 day of code restructuring. On a 4-day timeline, this is prohibitive.
- Stay with Qwen2.5-VL-7B and focus only on data/hyperparameters: Viable but leaves the largest lever (model upgrade) untouched. The 8B upgrade has a clean fallback path.
- Add 5+ external datasets: Rejected. Diminishing returns on data quantity. KVQA + TOD are the highest-quality, most relevant options with instant access.

**Why chosen:** The delta approach preserves the strong existing implementation while targeting the highest-ROI improvements. Each delta is gated by an A/B test, so no change can silently degrade performance. The model upgrade (Delta 1) is the single highest expected-value change, and the fallback path (stay with 7B) is zero-cost.

**Consequences:**
- Hour 0 VRAM test adds ~1h before any training
- 8B training is ~1.2x slower than 7B per step
- External data pipeline adds ~2-3h coding time
- A/B testing protocol means each delta takes a full training cycle to validate
- Realistic total gain over current notebook: +3-10pp (honest range)

**Follow-ups:**
- After Day 2 submission, leaderboard feedback calibrates remaining effort
- If Qwen3-VL-8B Korean performance is strong, consider increasing epochs further
- Monitor per-question-type accuracy to identify weakest areas for targeted improvement
