# VQA Competition TOP-1 Delta Plan v6.1 -- H100 80GB RALPLAN-DR

**Created:** 2026-04-02
**Revised:** 2026-04-02 (v6.1 -- Architect+Critic consensus: mean-logprob normalization, Day-1 model smoke test, AutoModel prerequisite, mid-epoch checkpoint, inference time budget)
**Deadline:** 2026-04-06 (4 days remaining)
**Hardware:** Colab H100 (80GB VRAM) -- removes ALL VRAM constraints
**Base:** `ssafy_vqa_qwen25vl_h100_topscore_notebook_colab.ipynb`
**Objective:** Push from current strong baseline to TOP-1 accuracy on SSAFY 15th AI Challenge

---

## RALPLAN-DR Summary

### Principles (5)

1. **Delta-only**: 기존 노트북은 이미 매우 정교함. 재작성 절대 금지. 변경 최소화.
2. **A/B gate 필수**: 모든 delta는 독립적 A/B 테스트 통과 시에만 채택 (>= 0.5pp 향상).
3. **H100 자원 최대 활용**: 80GB VRAM = BF16 full precision training + batch size 4-8 + 높은 resolution. 4-bit quantization 제거가 가장 저비용 고효과 변경.
4. **시간 ROI 우선**: 4일 남음. 기대값(gain x confidence / time) 순서로 실행.
5. **Safe submission 우선**: Day 2 저녁까지 반드시 첫 제출. 이후 개선은 보너스.

### Decision Drivers (Top 3)

1. **H100 80GB = quantization 불필요** -- `train_use_4bit=True`를 `False`로 바꾸면 BF16 full precision training. 이것만으로 +1-3pp 기대 (quantization loss 제거). 코드 변경 1줄.
2. **Batch size 1 -> 4~8 가능** -- effective batch size 32-64로 training stability 향상. gradient accumulation 줄여서 training 속도 ~4x 개선. 하루 6-8 full run 가능 = 더 많은 A/B 테스트.
3. **Letter-level logprob -> Full-text logprob** -- 현재 `candidates=["a","b","c","d"]` 단일 토큰 scoring. Full choice text scoring은 모델이 답변 내용을 직접 평가 = 더 정확한 ranking.

### Viable Options (3)

#### Option A: Conservative H100 Adaptation (RECOMMENDED)
- BF16 training + batch size 증가 + epoch 증가 + resolution 증가
- Qwen2.5-VL-7B 유지 (안정성 우선)
- Full-text logprob scoring 추가
- Ensemble weight optimization
- **Expected gain:** +3-6pp
- **Risk:** LOW -- 검증된 모델에 H100 자원만 추가 투입
- **Time:** 3 days (Day 4 = safety buffer)

#### Option B: Aggressive Model Upgrade
- Option A 전체 + Qwen3-VL-8B 모델 교체
- BF16로 8B 모델도 여유있게 로드 (80GB VRAM)
- Material expert adapter 추가 (3-adapter system)
- **Expected gain:** +5-10pp
- **Risk:** MEDIUM -- Qwen3-VL-8B Korean 성능 미검증, chat template 변경 가능
- **Time:** 4 days (no buffer)

#### Option C: Data-Heavy Approach
- Option A + KVQA/TOD 외부 데이터 대량 추가
- Dawid-Skene로 dev label quality 재평가
- Learned stacker (logistic regression) 대신 fixed weights
- **Expected gain:** +4-8pp
- **Risk:** MEDIUM-HIGH -- 외부 데이터 noise, pipeline 코딩 시간 소모
- **Time:** 4 days (data pipeline 코딩에 ~1일 소요)

### Option Evaluation

| Criteria | Option A | Option B | Option C |
|----------|----------|----------|----------|
| Expected gain | +3-6pp | +5-10pp | +4-8pp |
| Implementation risk | LOW | MEDIUM | MEDIUM-HIGH |
| Time to first submission | Day 1 저녁 | Day 2 저녁 | Day 2 저녁 |
| A/B test 횟수 가능 | 8-10회 | 5-6회 | 4-5회 |
| Fallback cost | 0 (모든 변경 독립) | ~4h (model revert) | ~6h (data pipeline 폐기) |

**Decision: Option A를 base로 하되, Option B의 Qwen3-VL-8B upgrade를 Day 2에 conditional로 시도.**

---

## Reference: ChatGPT Qwen3-VL H100 Notebook (`ssafy_vqa_qwen3vl_colab_h100.ipynb`)

ChatGPT Pro가 작성한 Qwen3-VL-8B 전용 H100 노트북이 존재. **통째로 쓰지 않고 cherry-pick만 한다** — 우리 노트북의 핵심 기능(per-sample weighted loss, validation split, JPEG augmentation, merger unfreezing, dynamic resolution)이 ChatGPT 노트북에는 없기 때문.

### Cherry-Pick 항목 (반영 완료)
1. ✅ `Qwen3VLForConditionalGeneration` import 확인 (Delta 4 prerequisite에 반영)
2. ✅ `flash_attention_2` + SDPA fallback (Delta 4에 반영)
3. ✅ LoRA r=64, alpha=128 A/B test 후보 (Delta 2d에 반영)
4. ✅ OCR upsample 3x (Delta 4에 반영)
5. ✅ BF16에서 LR 조정 필요 (Delta 2 주의사항에 반영)

### Cherry-Pick 안 함 (우리 구현이 더 좋음)
- ❌ 4-bit quantization on H100 — BF16이 우월
- ❌ Standard loss (no per-sample weight) — 우리의 weighted loss가 외부/pseudo 데이터에 더 적합
- ❌ No validation split — 우리는 stratified validation 보유
- ❌ No JPEG augmentation — 우리는 있음
- ❌ No merger unfreezing — 우리는 있음
- ❌ batch_size=2 — H100 BF16에서는 4-8 가능

### Blend Weight 비교 (A/B 테스트 후보)
| Question Type | 우리 (shared/count/text/ocr) | ChatGPT (shared/count/text/ocr) |
|---------------|------------------------------|--------------------------------|
| count | 0.22 / 0.68 / 0.10 / 0.00 | 0.60 / 0.28 / 0.12 / 0.00 |
| material | 0.78 / 0.00 / 0.22 / 0.00 | 0.74 / 0.00 / 0.26 / 0.00 |
| object | 0.75 / 0.00 / 0.25 / 0.00 | 0.71 / 0.08 / 0.21 / 0.00 |
| color | 0.82 / 0.00 / 0.18 / 0.00 | 0.78 / 0.00 / 0.22 / 0.00 |
| location | 0.85 / 0.00 / 0.15 / 0.00 | 0.86 / 0.00 / 0.09 / 0.05 |
| ocr_text | 0.55 / 0.00 / 0.10 / 0.35 | 0.60 / 0.00 / 0.05 / 0.35 |

**핵심 차이: count 질문.** 우리는 count expert에 68% 가중치, ChatGPT는 shared에 60%. Delta 5a (Ensemble weight optimization)에서 validation 기반 최적화로 결정.

---

## CRITICAL: What Already Exists (DO NOT REWRITE)

기존 v5 plan의 "Already Implemented" 표 전체 유지. 아래 delta는 기존 노트북의 **CFG 값 변경 + 함수 추가**만 수행.

### Current Config (Baseline for All Deltas)
```python
# Colab notebook 현재 설정
model_id        = "Qwen/Qwen2.5-VL-7B-Instruct"
train_use_4bit  = True      # <-- H100에서 불필요. Delta 1에서 False로 변경
infer_use_4bit  = False     # <-- 이미 bf16 추론
train_dtype     = "bfloat16"
infer_dtype     = "bfloat16"
per_device_batch_size = 1   # <-- H100에서 4-8 가능. Delta 1에서 변경
grad_accum_steps      = 8   # <-- batch size 올리면 줄임
shared_epochs   = 1         # <-- Delta 2에서 증가
count_epochs    = 2         # <-- Delta 2에서 증가
train_min_pixels = 448*448
train_max_pixels = 1280*1280  # <-- Delta 1에서 1536*1536 가능
infer_max_pixels = 1536*1536  # <-- Delta 1에서 1792*1792 가능
n_perm_shared   = 2         # <-- Delta 2에서 증가
n_perm_count    = 3         # <-- Delta 2에서 증가
lora_r = 32, lora_alpha = 64, lora_dropout = 0.05
```

---

## Delta Plan: 5 Targeted Improvements

### ROI Assessment (Honest Expected Gains)

| Delta | Expected Gain | Confidence | Risk | Time Cost | H100 Dependency |
|-------|--------------|------------|------|-----------|-----------------|
| Delta 1: BF16 + Batch + Resolution | +1-3pp | HIGH | LOW | ~1h config | YES -- core unlock |
| Delta 2: Epoch + Scheduler + Permutations | +1-2pp | MEDIUM-HIGH | LOW | ~1h config + retrain | Partially (더 빠른 retrain) |
| Delta 3: Full-text logprob scoring | +1-2pp | MEDIUM | LOW | ~2h coding | YES (VRAM for larger batch inference) |
| Delta 4: Qwen3-VL-8B upgrade | +2-5pp | MEDIUM | MEDIUM | ~3h total | YES (BF16 8B = ~16GB, 4bit 불필요) |
| Delta 5: Ensemble optimization + Stretch | +0.5-1.5pp | MEDIUM-LOW | LOW | ~2h | No |
| **Combined realistic gain** | **+4-10pp** | | | | |

**Honesty note:** 기존 노트북은 이미 고도로 최적화됨. Delta 1(BF16 전환)이 가장 확실한 개선이고, Delta 4(모델 업그레이드)가 가장 큰 잠재력. 나머지는 marginal gains.

---

### Delta 1: H100 Full Utilization -- BF16 Training + Batch Size + Resolution (HIGHEST PRIORITY)

**Rationale:** 현재 노트북은 16GB VRAM 대상으로 4-bit quantization 사용 중. H100 80GB에서는 BF16 full precision이 가능하며, 이것만으로 quantization loss 제거 + 학습 안정성 향상.

**Code Changes (CFG 값 변경만):**

```python
# === Delta 1: H100 최적화 ===
# 변경 1: 4-bit quantization 제거 (1줄)
CFG.train_use_4bit = False  # was: True

# 변경 2: Batch size 증가 (2줄)
CFG.per_device_batch_size = 4   # was: 1
CFG.grad_accum_steps = 4        # was: 8  (effective batch = 4*4 = 16)
# Alternative: batch=8, accum=2 (effective=16) -- VRAM 확인 후 결정

# 변경 3: Training resolution 증가 (2줄)
CFG.train_min_pixels = 512 * 512    # was: 448*448
CFG.train_max_pixels = 1536 * 1536  # was: 1280*1280

# 변경 4: Inference resolution 증가 (2줄)
CFG.infer_min_pixels = 672 * 672    # was: 512*512
CFG.infer_max_pixels = 1792 * 1792  # was: 1536*1536
CFG.infer_ocr_max_pixels = 2048 * 2048  # was: 1792*1792
```

**VRAM Budget Estimate (BF16, Qwen2.5-VL-7B):**
- Model weights BF16: ~14GB
- LoRA trainable params: ~0.5GB
- Optimizer states (AdamW): ~1GB
- Activations (batch=4, 1536px): ~20-30GB
- Gradient checkpointing: -50% activation memory
- **Total estimate: ~30-45GB / 80GB available**
- Headroom: 35-50GB -- 충분

**Execution Protocol:**
1. `CFG.train_use_4bit = False` 설정 후 model load -- peak VRAM 확인
2. Batch size 4로 1 step forward+backward -- OOM 확인
3. OOM 없으면 batch size 8 시도 (effective batch = 32)
4. 최적 batch size 결정 후 full shared adapter training 실행
5. Training time 측정 (16GB 대비 얼마나 빨라졌는지)

**Go/No-Go:**
- Batch 4 + BF16 + 1536px training에서 OOM: batch 2로 fallback (여전히 BF16)
- Batch 2에서도 OOM: resolution을 1280px로 낮춤 (여전히 BF16)
- 어떤 경우에도 `train_use_4bit=True`로 돌아갈 이유 없음 (80GB VRAM)

**Acceptance Criteria:**
- [ ] `train_use_4bit=False`로 BF16 model load 성공, peak VRAM <= 50GB
- [ ] Batch size 4 이상에서 training 시작 성공
- [ ] Training speed: 16GB 4-bit baseline 대비 step time 비교 기록
- [ ] 1 epoch shared adapter 완료, validation accuracy 기록
- [ ] BF16 accuracy >= 4-bit accuracy (same epochs) -- 거의 확실하지만 확인 필수

**Estimated time:** ~30min config change + VRAM test + 1 epoch training (~45min at batch 4)

---

### Delta 2: Training Intensity -- Epochs + Scheduler + Permutations (HIGH PRIORITY)

**Rationale:** H100 batch size 증가로 training이 ~4x 빨라짐. 이 속도 이점을 epoch 수 증가에 투자. 기존 shared_epochs=1은 underfitting 가능성 있음.

**Code Changes:**

```python
# === Delta 2: Training Intensity ===
# 변경 1: Epoch 증가
CFG.shared_epochs = 2   # was: 1  (BF16 + batch 4 = ~1.5h for 2 epochs)
CFG.count_epochs = 3    # was: 2

# 변경 2: Cosine scheduler (import 1줄 + 함수 변경 1줄)
# get_linear_schedule_with_warmup -> get_cosine_schedule_with_warmup
# (같은 transformers 패키지에서 import)

# 변경 3: Inference permutation 증가
CFG.n_perm_shared = 3   # was: 2
CFG.n_perm_count = 5    # was: 3
# H100 추론 속도가 빠르므로 시간 부담 적음
```

**A/B Test Protocol:**
- **Test 2a:** shared_epochs=2 vs 1 (count_epochs 고정) -- underfitting 여부 확인
- **Test 2b:** cosine vs linear scheduler (best epochs 설정에서)
- **Test 2c:** n_perm 3/5 vs 2/3 (inference only, training 불필요)
- **Test 2d (ChatGPT 참고):** LoRA r=64, alpha=128 vs 현재 r=32, alpha=64 -- ChatGPT 노트북이 r=64 사용. 더 큰 capacity = 더 나은 task adaptation 가능. 단, BF16에서는 QLoRA보다 학습 안정적이므로 r=32로도 충분할 수 있음.
- 각 테스트 독립 실행. >= 0.5pp 향상 시에만 채택.

**⚠️ LR 주의사항 (BF16 vs QLoRA):**
ChatGPT 노트북은 QLoRA에서 lr_shared=2e-4를 사용. 우리는 BF16 (no quant)이므로 gradient가 더 정확하고 full precision → **LR을 낮춰야 함**. BF16에서는 5e-5 ~ 1e-4 범위가 적절. QLoRA용 2e-4는 BF16에서 발산 위험.

**Acceptance Criteria:**
- [ ] shared_epochs=2에서 validation accuracy 기록 (vs epochs=1)
- [ ] Cosine scheduler에서 validation accuracy 기록 (vs linear)
- [ ] n_perm 3/5에서 accuracy 기록 (vs 2/3)
- [ ] 각 A/B 결과 문서화, 0.5pp threshold 적용

**Estimated time:** ~2h (3 A/B tests, H100에서 각 ~40min)

---

### Delta 3: Full-Text Logprob Scoring (MEDIUM PRIORITY)

**Rationale:** 현재 `score_letters_with_logprob()`는 `candidates=["a","b","c","d"]` 단일 letter 토큰만 scoring. 모델이 "a"라는 글자의 확률만 보는 것. Full-text scoring은 실제 choice text (예: "유리", "3개")의 token-by-token log probability를 합산 -- 모델이 답변 내용을 직접 평가.

**Current Implementation (letter-only):**
```python
# Line 1274-1313: score_letters_with_logprob()
# candidates = ["a", "b", "c", "d"]  -- 단일 토큰
# full_texts = [prefix_text + c for c in candidates]
# scores = sum of log_probs for each single letter token
```

**Proposed Addition: score_fulltext_with_logprob()**

**CRITICAL: Length normalization (Architect+Critic consensus)**
Choice text 길이가 다르므로 (예: "유리" 2 tokens vs "플라스틱" 4+ tokens), sum-of-log-probs는 짧은 텍스트를 체계적으로 선호. **Mean-log-prob를 기본으로 사용하고, sum은 A/B variant로만 테스트.**

```python
def score_fulltext_with_logprob(
    model, processor, image, prefix_messages,
    choice_texts: List[str],  # ["유리", "금속", "종이", "플라스틱"]
    normalize: bool = True,    # DEFAULT: mean-log-prob (길이 정규화)
) -> np.ndarray:
    """
    Full choice text log-probability scoring.
    DEFAULT: score(choice_i) = (1/n_tokens) * sum_t log P(token_t | ...)
    normalize=False: score(choice_i) = sum_t log P(token_t | ...)
    """
    prefix_text = processor.apply_chat_template(
        prefix_messages, tokenize=False, add_generation_prompt=True
    )
    full_texts = [prefix_text + text for text in choice_texts]
    # ... (기존 score_letters_with_logprob와 동일한 패턴)
    # prefix_len 이후 token들의 log_prob 합산
    # normalize=True: gathered.sum() / len(target_ids)  ← DEFAULT
    # normalize=False: gathered.sum()
```

**Integration into ensemble:**
```python
# ensemble_weights에 "fulltext" channel 추가
# 또는 기존 letter logprob 대신 fulltext로 교체
# A/B test로 결정
```

**A/B Test:**
- Version A: Letter-only logprob (현재)
- Version B: Full-text logprob (letter logprob 대체)
- Version C: Letter + Full-text ensemble (두 채널 평균)
- Validation accuracy 비교. Best version 채택.

**Acceptance Criteria:**
- [ ] `score_fulltext_with_logprob()` 함수 구현, 기존 패턴 따름
- [ ] 단일 sample에서 letter vs fulltext score 비교 출력 (sanity check)
- [ ] A/B test: letter-only vs fulltext-only vs ensemble
- [ ] Best version 채택, accuracy delta 기록

**Estimated time:** ~2h coding + 1h A/B test

---

### Delta 4: Qwen3-VL-8B Model Upgrade (MEDIUM PRIORITY -- Conditional)

**Rationale:** v5 plan의 Delta 1과 동일하나, H100에서는 VRAM 걱정 불필요. BF16으로 8B 모델도 ~16GB로 로드. 진짜 risk는 Korean 성능만.

**H100에서의 변경 (v5 대비 간소화):**
- ~~VRAM smoke test~~ -> 불필요. 80GB에서 8B BF16은 여유
- ~~4-bit quantization 유지~~ -> 불필요. BF16 full precision
- ~~Resolution fallback~~ -> 불필요. 1536px training도 가능

**VRAM Budget (BF16, Qwen3-VL-8B):**
- Model weights BF16: ~16GB
- LoRA + Optimizer: ~1.5GB
- Activations (batch=4, 1536px, grad ckpt): ~25-35GB
- **Total: ~42-52GB / 80GB -- 충분**

**Prerequisite (Day 1에 미리 준비 -- Architect consensus + ChatGPT 노트북 검증):**

`load_base_model()` 함수에서 `Qwen2_5_VLForConditionalGeneration`이 하드코딩되어 있음. ChatGPT의 H100 노트북(`ssafy_vqa_qwen3vl_colab_h100.ipynb`)에서 **`Qwen3VLForConditionalGeneration`이 정상 작동 확인됨**.

두 가지 접근법:
- **Option A (간단):** `AutoModelForImageTextToText` 사용 — 모델 자동 resolve. 두 모델 모두 동작.
- **Option B (ChatGPT 검증):** `Qwen3VLForConditionalGeneration` 직접 import — ChatGPT 노트북에서 실제 동작 확인됨.

```python
# load_base_model() 변경:
# BEFORE:
#   from transformers import Qwen2_5_VLForConditionalGeneration
#   model = Qwen2_5_VLForConditionalGeneration.from_pretrained(CFG.model_id, **kwargs)
#
# AFTER (추천 — 7B에서도 8B에서도 동작):
from transformers import AutoModelForImageTextToText
model = AutoModelForImageTextToText.from_pretrained(CFG.model_id, **kwargs)
#
# ALTERNATIVE (ChatGPT 검증 — 8B 전용):
# from transformers import Qwen3VLForConditionalGeneration
# model = Qwen3VLForConditionalGeneration.from_pretrained(CFG.model_id, **kwargs)
```

**추가 (ChatGPT 노트북에서 검증된 설정):**

1. **Flash Attention 2 with SDPA fallback** (ChatGPT 노트북에서 확인):
```python
# CFG에 추가:
CFG.attn_implementation = "flash_attention_2"  # H100에서 최적

# auto fallback 함수:
def auto_choose_attn_impl(default="flash_attention_2"):
    if default == "flash_attention_2":
        try:
            import flash_attn
            return "flash_attention_2"
        except ImportError:
            return "sdpa"
    return "sdpa"

# model load 시:
model = AutoModelForImageTextToText.from_pretrained(
    CFG.model_id,
    attn_implementation=auto_choose_attn_impl(CFG.attn_implementation),
    **kwargs
)
```

2. **OCR question upsample** (ChatGPT 노트북에서 확인):
```python
CFG.upsample_ocr_text = 3   # OCR 질문을 3배 복제 (전체의 1.1%뿐이므로 underrepresented)
```

**Code Changes:**
1. `CFG.model_id = "Qwen/Qwen3-VL-8B-Instruct"` (1줄)
2. `attn_implementation` 추가 + auto fallback (flash_attn 설치 필요: `pip install flash-attn --no-build-isolation`)
3. `mm_token_type_ids` collator 통과 확인
4. Chat template + label masking 검증 (labels tensor 출력으로 확인)

**Execution Timeline (Architect consensus -- Day 1 smoke test 채택):**

> **Architect 권고 채택 이유:** H100에서 VRAM risk 제거됨. 유일한 risk는 Korean 성능이며, 40분 1-epoch smoke test로 Day 1에 검증 가능. 2일간 잘못된 모델을 최적화하는 것보다 40분 투자가 합리적.

- **Day 1 저녁 (Delta 1 완료 후):** Qwen3-VL-8B BF16 1-epoch smoke test (~40min)
  - Qwen2.5-VL-7B BF16 1-epoch accuracy와 비교
  - 8B accuracy >= 7B accuracy - 1pp: **GO** → Day 2부터 8B 기반으로 Delta 2/3 진행
  - 8B accuracy < 7B accuracy - 1pp: **NO-GO** → 7B 유지, Delta 4 취소
- **Day 2 safe submission:** 선택된 모델(8B or 7B)로 제출
- 이렇게 하면 Delta 2/3의 A/B test가 실제 제출할 모델에서 수행됨 → 결과가 더 유의미

**Acceptance Criteria:**
- [ ] Qwen3-VL-8B BF16 load 성공 (peak VRAM 기록)
- [ ] Chat template + label masking 검증 (labels tensor 출력 확인)
- [ ] 1 epoch shared adapter training 완료
- [ ] Validation accuracy: Qwen3-VL-8B vs Qwen2.5-VL-7B (동일 config)
- [ ] Go/No-Go decision 문서화

**Estimated time:** ~1h setup + 1.5h training + 30min validation = ~3h
**Fallback cost:** ~3h (Qwen2.5-VL-7B로 돌아가면 됨. 안전 제출 이미 완료 상태.)

---

### Delta 5: Ensemble Optimization + Stretch (LOWEST PRIORITY)

**Only attempt if Day 3 evening and Deltas 1-4 resolved.**

**5a: Ensemble Weight Optimization**
```python
# 현재: 수동 설정된 고정 weights
# ensemble_weights = {
#     "count":    {"shared": 0.22, "count": 0.68, "text": 0.10, "ocr": 0.00},
#     "material": {"shared": 0.78, "count": 0.00, "text": 0.22, "ocr": 0.00},
#     ...
# }

# 제안: Validation set에서 scipy.optimize 또는 grid search로 최적화
from scipy.optimize import minimize
# Per-qtype weight optimization on validation predictions
```
- Risk: Validation set overfitting (236 validation samples per qtype 정도)
- Mitigation: Leave-one-out CV 또는 regularization

**5b: Material Expert Adapter (Optional)**
- 현재: shared + count 2-adapter system
- 제안: shared + count + material 3-adapter system
- material questions이 33.4%로 두 번째로 많음
- Dedicated material adapter가 도움될 수 있음
- H100에서 추가 training time 부담 적음

**5c: Test-Time Augmentation (Horizontal Flip)**
- Non-OCR, non-text questions에만 적용
- 2x inference time이지만 H100에서는 부담 적음

**Acceptance Criteria:**
- [ ] 각 stretch technique 독립 A/B test
- [ ] >= 0.5pp 향상 시에만 채택

---

## Timeline (4 Days -- H100 Optimized)

H100 batch size 4-8에서 1 epoch shared ≈ 20-40min (vs 16GB에서 ~2.5h).
하루 6-8 full training run 가능 = 더 많은 A/B 실험.

| Day | Block | Activity | Deliverable |
|-----|-------|----------|-------------|
| **Day 1** | 오전 | **Delta 1:** BF16 전환 + batch size 증가 + resolution 증가 + `AutoModelForImageTextToText` 전환. VRAM test. | BF16 training 성공 확인 |
| | 오후 | **Delta 1 validation:** 7B BF16 1 epoch shared + 2 epoch count. Validation accuracy 기록. | BF16 baseline accuracy |
| | 저녁 | **Delta 4 smoke test:** Qwen3-VL-8B BF16 1-epoch. 7B vs 8B 비교 → **모델 결정 (Go/No-Go).** | Model decision (Day 1에 확정) |
| **Day 2** | 오전 | **Delta 2a/2b:** 선택된 모델에서 epochs A/B + cosine scheduler A/B. | Epoch/scheduler A/B results |
| | 오후 | **Delta 2c + Delta 3:** Permutation A/B + full-text logprob(mean-normalized) 구현 + A/B. | Logprob/perm A/B results |
| | 저녁 | **SAFE SUBMISSION #1** (Delta 1+2+3 best config, 선택된 모델). | First submission |
| **Day 3** | 오전 | **Delta 4 full training** (if 8B GO): 선택된 best hyperparams로 8B full training (2-3 epochs). | Full-trained model |
| | 오후 | **Delta 5a:** Ensemble weight optimization on validation set. | Optimized weights |
| | 저녁 | **SUBMISSION #2** (best model + optimized ensemble). | Second submission |
| **Day 4** | 오전 | **Delta 5b/5c:** Material expert adapter + TTA (stretch). | Stretch A/B results |
| | 오후 | Final tuning + best config retrain. | Final model |
| | 저녁 (마감) | **FINAL SUBMISSION**. Safety backup 30min before deadline. | Final submission |

### Time Budget (H100 기준)

| Activity | Estimated Time | Notes |
|----------|---------------|-------|
| Delta 1: BF16 config + VRAM test + AutoModel 변경 | 30min | CFG 변경 5줄 + load_base_model 수정 |
| Delta 1: Full training (shared 1ep + count 2ep) | 1.5h | Batch 4, BF16 |
| Delta 1 추가: Qwen3-VL-8B smoke test | 40min | Day 1 저녁, 1-epoch Go/No-Go |
| Delta 2: 3x A/B tests | 3h | 각 ~1h (training + eval) |
| Delta 3: Full-text logprob coding + A/B | 3h | 2h coding + 1h test |
| Delta 4: Qwen3-VL-8B full training (if GO) | 2h | BF16, batch 4-8 |
| Delta 5: Stretch techniques | 2h | If time remains |
| Submissions + buffer | 2h | 3 submissions + safety margin |
| **Total** | **~14.5-15h active** | 4 days에 충분 |

### Inference Wall-Clock Estimate (Critic consensus)

**현재 (n_perm_shared=2, n_perm_count=3):**
- Non-count sample: 2 perm × 1 forward = 2 forwards
- Count sample (33.8%): 2 shared + 3 count = 5 forwards
- Uncertain count (margin < threshold): 5 + 1 hi-res = 6 forwards
- H100 BF16 추론 ~0.3-0.5s/forward (1536px, batch=1 per forward)
- **예상 총 추론 시간:** 5,074 samples × avg ~3 forwards × 0.4s ≈ **~100min (~1.7h)**

**제안 (n_perm_shared=3, n_perm_count=5 + full-text logprob):**
- Non-count: letter 3 + fulltext 3 = 6 forwards
- Count: letter (3 shared + 5 count) + fulltext (3+5) = 16 forwards
- Uncertain count: 16 + 2 hi-res = 18 forwards
- **예상 총 추론 시간:** 5,074 × avg ~8 forwards × 0.4s ≈ **~270min (~4.5h)**

**⚠️ 4.5h는 너무 길다.** Day 2 safe submission 전에 inference만 4.5h가 걸리면 일정 위험.

**Mitigation:**
- Full-text logprob은 letter logprob보다 좋을 때만 채택 (A/B gate)
- 둘 중 하나만 사용 (letter OR fulltext, not both) → 추론 시간 원래 수준 유지
- Permutation 증가 (3/5)는 inference 시간 1.5x → ~150min (~2.5h). 허용 범위.
- **결론: letter+fulltext 동시 사용은 Day 4 stretch에서만. Day 2 safe submission은 단일 scoring 방식으로.**

---

## Risk Mitigation (H100 Adapted)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ~~Qwen3-VL-8B OOM~~ | ~~MEDIUM-HIGH~~ | ~~HIGH~~ | **ELIMINATED** -- 80GB VRAM에서 BF16 8B는 ~52GB |
| Colab H100 session disconnection | MEDIUM | HIGH | **Mid-epoch checkpoint (Critic consensus):** `train_one_adapter()` 내부에 매 200 optimizer steps마다 adapter 저장 추가. `if global_step % 200 == 0: model.save_pretrained(save_dir / f"step_{global_step}")`. H100 batch=4 기준 200 steps ≈ 5-10분. 최악의 경우 10분 분량만 손실. Google Drive에 자동 sync: `shutil.copytree(save_dir, f"/content/drive/MyDrive/vqa_checkpoints/{adapter_name}")`. |
| Colab GPU quota 소진 | LOW-MEDIUM | HIGH | 유료 Colab Pro/Pro+ 확인. Session 재연결 시 checkpoint에서 resume. |
| BF16 training이 4-bit 대비 큰 차이 없음 | LOW | LOW | 이미 첫 A/B에서 확인. 차이 없어도 training speed 향상은 유지. |
| Qwen3-VL-8B Korean 성능 약함 | LOW-MEDIUM | MEDIUM | Day 2 안전 제출 후 시도. Revert cost = ~3h. |
| Full-text logprob이 letter logprob보다 안 좋음 | LOW-MEDIUM | LOW | A/B test. Letter로 fallback. |
| Epoch 증가로 overfitting | MEDIUM | LOW | Validation loss monitoring. Per-epoch checkpoint에서 best epoch 선택. |
| 제출 횟수 제한으로 A/B 결과 반영 불가 | LOW | MEDIUM | Validation set accuracy 기준으로 결정. 제출은 최종 best만. |

---

## Decision Tree (H100 Adapted)

```
Day 1 오전-오후: Delta 1 -- BF16 Full Precision + AutoModel 전환
  |
  +-- BF16 batch=4 성공 (거의 확실)
  |     |
  |     +-- batch=8 시도
  |     |     +-- 성공: batch=8, accum=2 사용
  |     |     +-- OOM: batch=4, accum=4 사용
  |     |
  |     +-- 7B BF16 1 epoch training 완료, baseline accuracy 기록
  |
  +-- Day 1 저녁: Delta 4 smoke test (40min)
        |
        +-- 8B accuracy >= 7B - 1pp: **GO 8B** → Day 2부터 모든 A/B를 8B 기반으로
        +-- 8B accuracy < 7B - 1pp: **NO-GO** → 7B 유지, Delta 4 취소
Day 2: Delta 2 + Delta 3 (선택된 모델 기반)
  |
  +-- Delta 2a: epochs A/B (선택된 모델에서)
  |     +-- epochs=2 > epochs=1 + 0.5pp: KEEP 2 epochs
  |     +-- epochs=2 <= epochs=1 + 0.5pp: KEEP 1 epoch
  +-- Delta 2b: Cosine scheduler A/B
  +-- Delta 2c: Permutation count A/B
  +-- Delta 3: Full-text logprob A/B (mean-normalized)
  |
  +-- Day 2 저녁: SAFE SUBMISSION #1 (Delta 1+2+3 best config, 선택된 모델)

Day 3: Delta 4 full training (if 8B GO) + Delta 5
  |
  +-- 8B GO: Best hyperparams로 full training (2-3 epochs) -> SUBMISSION #2
  +-- 8B NO-GO: 7B로 Delta 5 진행 -> SUBMISSION #2
  +-- Delta 5a: Ensemble weight optimization

Day 4: Final
  |
  +-- Delta 5b/5c: Material expert / TTA (stretch)
  +-- Final tuning + best config retrain
  +-- FINAL SUBMISSION (deadline 30min 전)
```

---

## ADR (Architecture Decision Record)

**Decision:** H100 80GB VRAM을 활용하여 기존 정교한 노트북에 5개 targeted delta를 적용. (1) BF16 full precision + batch/resolution 증가, (2) epoch/scheduler/permutation 최적화, (3) full-text logprob scoring, (4) conditional Qwen3-VL-8B upgrade, (5) ensemble optimization + stretch.

**Drivers:**
1. H100 80GB는 모든 VRAM 제약을 제거 -- 4-bit quantization이 더 이상 필요하지 않으며 이를 제거하는 것이 가장 확실한 품질 향상
2. Batch size 증가 = training speed ~4x -> 하루 6-8 full run = 더 많은 A/B experiment -> 더 나은 final config
3. 기존 노트북의 letter-level logprob scoring은 보수적; full-text scoring이 더 정확한 choice ranking 제공
4. Qwen3-VL-8B의 benchmark 우위는 크지만 Korean 성능이 미검증이므로 conditional로 처리
5. 4일 deadline에서 Day 2 safe submission이 최우선

**Alternatives considered:**
- **v5 plan 그대로 실행 (16GB 기준):** Rejected. 4-bit quantization 유지 = H100의 가장 큰 이점(full precision) 미활용. Batch size 1 유지 = 80GB 중 ~10GB만 사용.
- **Complete notebook rewrite for H100:** Rejected. 기존 노트북이 이미 정교하며, BF16 전환은 CFG 1줄 변경으로 충분.
- **InternVL3-8B or other model family:** Rejected. Processor API 변경으로 ~1일 소요. Qwen3-VL-8B가 같은 Qwen family라 코드 변경 최소.
- **Focus only on data (KVQA/TOD/AI Hub):** Rejected. 외부 데이터 pipeline 코딩에 ~1일 소요. H100에서 BF16 전환이 같은 시간에 더 확실한 gain.
- **Option C (Data-Heavy):** Not chosen as primary. KVQA/TOD 외부 데이터는 format 불확실성 + noise 리스크. Delta 1-3의 확실한 gain을 먼저 확보.

**Why chosen:** H100 전환의 가장 큰 이점은 (1) quantization 제거, (2) training speed 증가 두 가지. 이를 최우선으로 활용하고, 속도 이점으로 확보한 시간을 A/B testing에 투자. Qwen3-VL-8B는 safe submission 후 conditional 시도로 downside 제한.

**Consequences:**
- Day 1에 BF16 baseline 확립 후 빠른 iteration 가능
- Day 2 safe submission으로 최소 "strong baseline" 보장
- Qwen3-VL-8B 시도가 Day 3로 밀리지만, safe submission이 이미 있으므로 risk 최소
- Full-text logprob 코딩에 ~2h 소요하지만, accuracy gain 잠재력이 높음
- 외부 데이터(KVQA/TOD)는 이 plan에서 제외 -- 시간 ROI가 낮음

**Follow-ups:**
- Day 2 submission 후 leaderboard score 확인 -> 남은 gap 크기에 따라 Day 3-4 전략 조정
- Per-question-type accuracy breakdown으로 weakest area 식별 -> targeted improvement
- Qwen3-VL-8B가 성공하면 epoch/scheduler도 re-tune 필요 (optimal config이 다를 수 있음)
- 여유 시간에 KVQA/TOD 외부 데이터 시도 가능 (Delta 5 이후)
