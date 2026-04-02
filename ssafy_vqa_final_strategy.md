# SSAFY 재활용품 VQA 최종 전략 (Plan v7 -- Claude+ChatGPT 통합)

작성 목적: 7차례 plan iteration (Planner→Architect→Critic consensus), ChatGPT Pro 전략, Competition Discussion 인사이트, 그리고 실제 코드 분석을 통합한 **실전 최종 전략**.

---

## 1. 한 줄 결론

> **Qwen3.5-9B BF16 + EDA-driven 데이터 최적화 + 24-type 전문가 라우팅 + leaderboard 20회/일 A/B testing**

가장 승률이 높은 조합:

1. **메인 모델**
   - **1순위:** `Qwen/Qwen3.5-9B` (네이티브 멀티모달, early fusion)
   - **Fallback:** `Qwen/Qwen3-VL-8B-Instruct` (검증된 VL 구조)
   - **Last resort:** `Qwen/Qwen2.5-VL-7B-Instruct` (현재 노트북 기준)
   - Day 1 3-model smoke test → 승자로 4일 올인

2. **학습 전략**
   - H100 80GB: **BF16 full precision** (4-bit QLoRA 아님)
   - Batch 4-8, cosine scheduler, LoRA r=64/alpha=128
   - **EDA-first:** 먼저 돌리고 → per-type 약점 식별 → targeted 강화

3. **전문가 분업**
   - `shared adapter` + `count expert adapter` (기존)
   - 24개 세부 질문 유형별 특화 prompt + blend weight

4. **추론 전략**
   - letter logprob reranking (primary)
   - mean-normalized full-text logprob (A/B 테스트 후 채택 여부)
   - option permutation averaging (shared 3회, count 5회)
   - text prior + OCR + count 보조 feature 합성

5. **데이터 전략**
   - **train 5,073** + **dev pseudo label (confidence gradient)** + **TrashNet/TACO synthetic**
   - AI Hub 생활 폐기물은 **EDA에서 material_general이 약할 때만 조건부**
   - 20GB 맹목 다운로드 금지 → 기존 데이터 최적화가 시간 ROI 더 높음

---

## 2. 검증된 데이터 분석 (실제 CSV 기반)

### 2.1 broad q_type 비중 (우리 EDA 검증)

| split | count | material | object | color | location | ocr | 합계 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train (5,073) | 34.4 | 33.4 | 19.8 | 8.4 | 2.9 | 1.2 | 100 |
| dev (4,413) | **71.4** | 13.0 | 8.7 | 3.4 | 3.0 | 0.5 | 100 |
| test (5,074) | 33.8 | 32.1 | 22.1 | 8.2 | 2.8 | 1.1 | 100 |

**핵심:** dev는 71.4%가 count → train/test와 분포가 완전히 다름.
dev를 전체 모델에 강하게 섞으면 **material/object 성능 하락**. count expert에만 강하게 반영.

### 2.2 detail q_type 상위 (Discussion 그래프 + 우리 검증)

| 순위 | train | test | dev |
|---|---|---|---|
| 1 | material_general 23.1% | material_general 22.6% | count_general 21.2% |
| 2 | object_type 16.7% | object_type 18.8% | count_bottle 18.6% |
| 3 | count_bottle 9.6% | count_bottle 10.1% | count_box 13.5% |
| 4 | count_general 7.5% | count_general 8.0% | count_cup 11.6% |
| 5 | count_cup 6.7% | count_cup 6.8% | material_general 8.7% |

**→ Test Top 5 = 66.3%.** 이 5개 유형을 지배하면 상위권 확정.

### 2.3 텍스트 편향 분석 (ChatGPT 검증 + 우리 추가 분석)

- Text-only (TF-IDF/SGD) 5-fold OOF: **~45.2%**
- 하지만 같은 질문 텍스트라도 **89%는 다른 이미지에서 다른 정답** → 텍스트만으로는 한계
- material/object/recycle에서 텍스트 편향 강함, count에서는 약함
- **결론:** text prior는 material/object에 가중치 높게, count에 낮게

### 2.4 질문 반복률

- test 질문의 **64.4%가 train에 동일 문자열 존재**
- test의 **23.3%는 train에 없는 새로운 선택지 포함**
- count/material은 반복률 높음 (70%+), location/brand는 낮음 (17-28%)

### 2.5 Dev 합의도

| 합의 수준 | 샘플 수 | 비율 | 추정 정확도 |
|---|---:|---:|---|
| 5/5 일치 | 0 | 0% | - |
| 4/5 일치 | 902 | 20.4% | ~95%+ |
| 3/5 일치 | 2,606 | 59.1% | ~42-75% |
| 2/5 이하 | 904 | 20.5% | 노이즈 → 버림 |

**4/5 일치 902개:** 79.3%가 counting → count expert 강화용으로 최적
**d 편향:** 고신뢰 샘플 중 d=45.8% → answer-label 층화 없이 쓰면 d 편향 오염

### 2.6 이미지 특성

- 평균: 가로 ~760px, 세로 ~940px
- **80.6%가 세로형** (핸드폰 사진)
- 384x384 강제 stretch = 형태 왜곡 → **dynamic resolution 필수** (이미 구현됨)

---

## 3. 모델 전략

### 3.1 Day 1 3-Model Smoke Test

| 모델 | 파라미터 | H100 BF16 VRAM | 장점 | 리스크 |
|---|---|---|---|---|
| **Qwen3.5-9B** | 9B dense | ~18GB (여유) | 네이티브 멀티모달, 최신 벤치 상위, 201개 언어 | Korean 성능 미검증 |
| Qwen3-VL-8B | ~9B | ~16GB (여유) | 검증된 VL, OCR 강함, 공간 이해 | Qwen3.5보다 이전 세대 |
| Qwen2.5-VL-7B | 7B | ~14GB (여유) | 현재 노트북 baseline, KOFFVQA 67.7 검증 | 가장 오래된 모델 |

**Protocol:**
1. 각 모델 BF16 + LoRA 1-epoch 학습 (~40min each on H100)
2. Validation accuracy 비교
3. **3개 모두 leaderboard 제출** (20회/일 활용)
4. 승자 = PRIMARY MODEL → 이후 모든 A/B를 이 모델에서 진행

**Fallback chain:** Qwen3.5-9B → Qwen3-VL-8B → Qwen2.5-VL-7B

### 3.2 H100 BF16 학습 설정 (QLoRA가 아닌 이유)

| 설정 | QLoRA (ChatGPT 방식) | BF16 LoRA (우리 방식) |
|---|---|---|
| 가중치 정밀도 | 4-bit NF4 | **BF16 full** |
| 양자화 손실 | 있음 | **없음** |
| H100 VRAM 활용 | ~10GB (70GB 낭비) | **~45-55GB (적절)** |
| Batch size | 2 | **4-8** |
| 학습 안정성 | 양자화 noise | **더 안정적** |
| LR 범위 | 1e-4 ~ 2e-4 | **5e-5 ~ 1e-4** (낮춰야 함) |

**핵심:** H100 80GB에서 4-bit 양자화는 자원 낭비. BF16이 품질+속도 모두 우월.

### 3.3 최종 학습 하이퍼파라미터

```python
# 노트북 적용 완료 (ssafy_vqa_qwen3vl_h100_topscore.ipynb)
model_id          = "Qwen/Qwen3.5-9B"      # Day 1 smoke test 후 확정
train_use_4bit    = False                    # BF16 full precision
attn_implementation = "sdpa"                 # flash-attn 설치 불필요
per_device_batch  = 4                        # H100 BF16
grad_accum        = 4                        # effective batch = 16
shared_epochs     = 2                        # H100 속도로 가능
count_epochs      = 3
lr_shared         = 5e-5                     # BF16 적합
lr_count          = 7e-5
lora_r            = 64                       # 높은 capacity
lora_alpha        = 128
scheduler         = cosine with warmup
train_min_pixels  = 512 * 512
train_max_pixels  = 1536 * 1536
infer_min_pixels  = 672 * 672
infer_max_pixels  = 1792 * 1792
infer_ocr_max     = 2048 * 2048
n_perm_shared     = 3
n_perm_count      = 5
```

---

## 4. 데이터 전략: EDA-First (맹목 다운로드 금지)

### 4.1 원칙

> **20GB 외부 데이터를 맹목적으로 넣지 말고, 먼저 모델을 돌려서 어디가 약한지 파악하라.**

기존 데이터(train 5,073 + dev pseudo + TrashNet + TACO)만으로도 강한 baseline 확보 가능.
EDA 결과 특정 유형이 약할 때만 해당 유형에 targeted 외부 데이터 추가.

### 4.2 데이터 소스 우선순위

| 소스 | 크기 | 다운로드 | 용도 | 우선순위 |
|---|---|---|---|---|
| **Train (대회)** | 5,073 | 이미 있음 | 핵심 학습 데이터 | **필수** |
| **Dev pseudo label** | ~3,500 usable | 이미 있음 | count expert + 약점 보강 | **필수** |
| **TrashNet** | 2,527 | 코드 자동 (datasets) | material 보조 | **기존 구현** |
| **TACO** | 1,500 | git clone | count synthetic | **기존 구현** |
| AI Hub 생활 폐기물 | 150K+ (20GB+) | 수동 (aihub.or.kr) | material_general 강화 | **조건부** |

### 4.3 Dev 데이터 최적화 (기존 데이터 더 잘 활용)

현재 dev 활용이 binary threshold (0.80/0.60). 더 세밀하게 조정:

```python
# 현재: 고정 가중치
dev_shared_weight = 0.65  # 4/5 일치 → 모두 0.65

# 개선: confidence gradient
# 5/5 일치 → weight 0.95 (없음, 하지만 구조상 대비)
# 4/5 일치 → weight 0.75
# 3/5 일치 (count만) → weight 0.40
# 3/5 non-count → EDA 결과에 따라 조건부 (weight 0.35)
```

### 4.4 AI Hub 조건부 사용

```
Day 1: 모델 smoke test + baseline training
Day 1 저녁: Per-type EDA → 24개 유형별 accuracy breakdown
         ↓
IF material_general accuracy < 전체 평균 - 5pp:
    → AI Hub 소수 카테고리만 다운로드 (~2-3GB)
    → material 특화 synthetic VQA 생성
ELSE:
    → Skip. Dev + TrashNet/TACO로 충분.
```

---

## 5. 전문가 구조: Shared + Count Expert + 24-Type Router

### 5.1 기존 2-adapter 구조 (이미 구현)

| Adapter | 역할 | 데이터 | Epochs |
|---|---|---|---|
| **shared** | 전체 qtype 공통 기반 | train + dev pseudo + TrashNet + TACO | 2 |
| **count expert** | counting 특화 | train count + dev count pseudo + TACO count | 3 |

### 5.2 24-type 세부 라우터 (신규)

Discussion에서 확인된 24개 세부 유형으로 라우터 확장:

```python
DETAIL_TYPES = {
    # count 세부 (test ~34%)
    "count_bottle", "count_general", "count_cup", "count_box",
    "count_can", "count_lid",
    # material 세부 (test ~32%)
    "material_general", "material_package", "material_cup",
    "material_bottle", "material_can",
    # object (test ~22%)
    "object_type", "majority_type", "majority_general",
    # color 세부 (test ~8%)
    "color_lid", "color_straw", "color_general", "color_label",
    # ocr/brand (test ~1%)
    "brand_or_product", "text_or_symbol", "flavor",
    # other (test ~3%)
    "recycling_class", "position", "other",
}
```

**핵심:** Top 5 유형에 특화된 system prompt:

```python
SYSTEM_PROMPTS_DETAIL = {
    "material_general": (
        "너는 재활용 소재 전문가다. 유리, 플라스틱, 금속, 종이, 스티로폼 등 "
        "재질을 정확히 판별한다. 물체의 표면 질감, 투명도, 광택을 주의 깊게 본다."
    ),
    "count_bottle": (
        "너는 병 개수 세기 전문가다. 플라스틱병, 유리병, 페트병을 구분하여 정확히 센다. "
        "겹쳐 있거나 부분적으로 보이는 병도 포함한다."
    ),
    "count_cup": (
        "너는 컵 개수 세기 전문가다. 종이컵, 플라스틱컵, 텀블러를 구분하여 센다. "
        "뚜껑이나 빨대는 컵 개수에 포함하지 않는다."
    ),
    "object_type": (
        "너는 재활용품 종류 식별 전문가다. 이미지의 주요 물체가 어떤 종류의 "
        "재활용품인지 정확히 판별한다."
    ),
    # ... 나머지 유형별 prompt
}
```

### 5.3 유형별 앙상블 가중치

현재 7-type 가중치 → 24-type으로 세분화 가능 (Day 3에서 validation 기반 최적화).

**현재 기본값 (검증 기준):**

| q_type | shared | count | text | ocr |
|---|---:|---:|---:|---:|
| count_* | 0.22 | 0.68 | 0.10 | 0.00 |
| material_* | 0.78 | 0.00 | 0.22 | 0.00 |
| object_* | 0.75 | 0.00 | 0.25 | 0.00 |
| color_* | 0.82 | 0.00 | 0.18 | 0.00 |
| location | 0.85 | 0.00 | 0.15 | 0.00 |
| ocr/brand | 0.55 | 0.00 | 0.10 | 0.35 |

**Day 3에서 scipy.optimize 또는 grid search로 validation 기반 최적화.**

---

## 6. 추론 전략

### 6.1 Letter logprob reranking (Primary -- 이미 구현)

```python
# 각 선택지 a/b/c/d에 대한 next-token logprob 비교
# 자유 생성 + 파싱이 아니라 확률 직접 비교 → 파싱 에러 0%
candidates = ["a", "b", "c", "d"]
score(choice_i) = log P(token_i | image, prompt)
answer = argmax(scores)
```

### 6.2 Full-text logprob (A/B 테스트 후 채택 여부)

```python
# 선택지 텍스트 전체의 token-by-token log probability
# CRITICAL: 길이 정규화 필수 (Architect+Critic consensus)
choice_texts = ["유리", "금속", "종이", "플라스틱"]
score(choice_i) = (1/n_tokens) * sum_t log P(token_t | image, prompt, tokens_<t)
# normalize=True (DEFAULT) → 길이 편향 제거
```

**주의:** "유리" (2 tokens) vs "플라스틱" (4+ tokens) → sum만 쓰면 짧은 텍스트 선호.
반드시 mean-log-prob 사용.

### 6.3 Option permutation averaging

```python
# 보기 순서를 3-5회 셔플하여 재평가
# 원래 letter 위치로 환원 후 평균
# letter-position bias 제거 → 안정적으로 +0.5-1.5pp
n_perm_shared = 3  # 일반 질문
n_perm_count  = 5  # counting 질문 (더 불확실)
```

### 6.4 Text prior branch

```python
# 입력: question + a/b/c/d 텍스트만 (이미지 없이)
# 모델: char/word TF-IDF + SGD classifier
# 출력: 4-class probability

# material/object에서 text prior 가중치 높게 (텍스트 편향 강함)
# count에서는 가중치 낮게 (비전 정보가 더 중요)
```

### 6.5 불확실 샘플 2차 추론

```python
# 조건: top1 - top2 margin < threshold (0.12)
# 대상: count, location, brand, OCR 유형
# 수행:
#   - 고해상도 재추론 (infer_ocr_max_pixels = 2048*2048)
#   - 추가 permutation
#   - OCR engine 결과 반영
```

### 6.6 추론 시간 예산 (H100 기준)

```
현재 (n_perm 3/5, single scoring): ~150min (~2.5h) for 5,074 samples
→ Day 2 safe submission 전에 충분
→ letter + fulltext 동시 사용은 ~4.5h → Day 4 stretch에서만
```

---

## 7. 기존 노트북에 이미 구현된 것 (DO NOT REWRITE)

| Feature | 상태 | 구현 품질 |
|---|---|---|
| Label masking (answer-only loss, -100) | ✅ | prefix + padding 모두 마스킹 |
| Per-sample weighted loss | ✅ | source별 가중치 차등 |
| Full training data (5,073) | ✅ | answer-balanced |
| Dev pseudo labels | ✅ | shared >=0.80, count >=0.60 |
| TrashNet + TACO synthetic | ✅ | 2,500 + 1,800 max |
| Dual system prompt | ✅ | GENERAL + COUNT |
| Option shuffle augmentation | ✅ | 학습 시 a/b/c/d 셔플 |
| Shared + Count expert adapter | ✅ | 2-adapter 분리 학습 |
| Logprob reranking | ✅ | permutation 기반 |
| OCR (EasyOCR + rapidfuzz) | ✅ | OCR ensemble channel |
| Type-aware ensemble weights | ✅ | 7 types x 4 channels |
| Dynamic resolution | ✅ | min/max pixels |
| JPEG augmentation | ✅ | prob=0.35 |
| Stratified validation | ✅ | qtype + answer 층화 |
| Merger module unfreezing | ✅ | domain adaptation |
| Gradient checkpointing | ✅ | use_reentrant=False |
| Text-only TF-IDF prior | ✅ | predict_text_proba() |
| Mid-epoch checkpoint | ✅ | 200 steps + Drive sync |

---

## 8. 노트북에 새로 반영된 Delta 변경사항

| Delta | 변경 | 기대 효과 |
|---|---|---|
| **BF16 전환** | `train_use_4bit=False` | +1-3pp (양자화 손실 제거) |
| **Batch 4** | `per_device_batch=4, grad_accum=4` | 학습 4x 빠름 |
| **모델 업그레이드** | `Qwen/Qwen3.5-9B` | +3-8pp (Day 1 smoke test) |
| **LoRA r=64** | `lora_r=64, lora_alpha=128` | 더 큰 capacity |
| **Cosine scheduler** | `get_cosine_schedule_with_warmup` | 더 나은 수렴 |
| **Resolution 증가** | train 512-1536, infer 672-1792 | 세밀한 물체 인식 |
| **Permutation 증가** | shared 3, count 5 | position bias 감소 |
| **OCR upsample 3x** | `upsample_ocr_text=3` | 1.1% → 3.3% 비중 |
| **LR 조정** | `lr_shared=5e-5, lr_count=7e-5` | BF16 안정 학습 |
| **SDPA** | `attn_implementation="sdpa"` | flash-attn 설치 불필요 |
| **Google Drive** | `extract_dir="/content/drive/MyDrive"` | HF 다운로드 삭제 |
| **Mid-epoch ckpt** | 200 steps마다 + Drive sync | Colab 단절 보호 |
| **AutoModel** | `AutoModelForImageTextToText` | 모델 교체 1줄로 가능 |

---

## 9. 4일 실전 플랜 (20 submissions/day 활용)

### Day 1: 모델 결정 + Baseline 확보

| Block | Activity | 제출 |
|---|---|---|
| 오전 | **3-model smoke test:** Qwen3.5-9B vs Qwen3-VL-8B vs Qwen2.5-VL-7B (각 1-epoch ~40min) | #1-3 |
| 오후 | 승자 모델로 full training (shared 2ep + count 3ep) | #4-5 |
| 저녁 | **Per-type EDA:** 24개 유형별 accuracy breakdown → 약점 식별 | - |

**Day 1 Gate:** 모델 확정 + baseline leaderboard 점수 확인

### Day 2: A/B Testing + Dev 최적화

| Block | Activity | 제출 |
|---|---|---|
| 오전 | **A/B tests:** LoRA r=64 vs r=32, cosine vs linear scheduler | #6-8 |
| 오후 | **Dev 최적화:** confidence gradient weight, 조건부 >=3/5 추가 | #9-11 |
| 저녁 | EDA 약점 기반 targeted 개선. (IF material 약하면 → AI Hub 시작) | - |

**Day 2 Gate:** A/B 승자 확정 + dev 최적화 효과 확인

### Day 3: 24-Type Router + Logprob + Ensemble

| Block | Activity | 제출 |
|---|---|---|
| 오전 | **24-type router** 구현 + Top 5 유형별 특화 prompt | #12-13 |
| 오후 | **Full-text logprob** (mean-normalized) A/B + ensemble weight optimization | #14-16 |
| 저녁 | Best config 최종 확정 | - |

**Day 3 Gate:** best config 확정

### Day 4: Final Ensemble + Safety

| Block | Activity | 제출 |
|---|---|---|
| 오전 | Best config final retrain + 2nd model (seed=123) for ensemble | - |
| 오후 | 2-model ensemble + permutation inference | #17-19 |
| 저녁 | **FINAL SUBMISSION** (deadline 30min 전 safety backup) | **#20** |

### 제출 예산 (4일 x 20회 = 80회 총 가용)

| Day | 예산 | 용도 |
|---|---|---|
| Day 1 | 5회 | 3-model smoke test + baseline variants |
| Day 2 | 6회 | A/B winners + dev 최적화 |
| Day 3 | 5회 | 24-type router + logprob + ensemble weight |
| Day 4 | 4회 | Final ensemble + safety backups |

---

## 10. Decision Tree

```
Day 1 오전: 3-Model Smoke Test
  │
  ├── Qwen3.5-9B → val acc A, leaderboard score A'
  ├── Qwen3-VL-8B → val acc B, leaderboard score B'
  └── Qwen2.5-VL-7B → val acc C, leaderboard score C'
  │
  └── Winner = max(A', B', C') → PRIMARY
      │
      Day 1 오후: Full training with PRIMARY
      │
      Day 1 저녁: Per-type EDA
        ├── material_general acc < avg-5pp → AI Hub 조건부 다운로드
        └── All types OK → Dev + TrashNet/TACO로 진행

Day 2: A/B Tests (leaderboard 검증)
  │
  ├── LoRA r=64 vs r=32 → leaderboard 비교
  ├── Cosine vs Linear → leaderboard 비교
  ├── Dev confidence gradient → leaderboard 비교
  │
  └── A/B 승자 확정

Day 3: 24-Type + Logprob + Ensemble
  │
  ├── 24-type router + 특화 prompt → leaderboard
  ├── Full-text logprob (mean-norm) → leaderboard
  └── Ensemble weight optimization → best config 확정

Day 4: Final
  │
  ├── Best config retrain + 2nd model (seed=123)
  ├── 2-model ensemble + permutation inference
  └── FINAL SUBMISSION (deadline 30min 전)
```

---

## 11. 리스크 관리

| 리스크 | 확률 | 영향 | 대응 |
|---|---|---|---|
| Qwen3.5-9B Korean 성능 약함 | LOW-MED | HIGH | Day 1 smoke test에서 즉시 감지. Fallback: Qwen3-VL-8B |
| Colab H100 세션 단절 | MED | HIGH | 200 step마다 checkpoint + Drive sync. 최악 10분 손실 |
| BF16 batch=4 OOM | LOW | LOW | batch=2 fallback (여전히 BF16) |
| Full-text logprob이 letter보다 안 좋음 | MED | LOW | A/B gate. Letter로 fallback |
| Dev pseudo label이 오히려 해침 | LOW-MED | MED | A/B gate. 제거 fallback |
| 24-type router가 6-type보다 안 좋음 | LOW | LOW | A/B gate. 6-type 유지 |
| AI Hub 데이터 승인 지연 | MED | LOW | 조건부이므로 기본 전략에 영향 없음 |

---

## 12. 우선순위 체크리스트

### 무조건 할 것 (Day 1)
- [ ] 3-model smoke test (Qwen3.5-9B vs 3-VL-8B vs 2.5-VL-7B)
- [ ] BF16 전환 + batch 4 + 고해상도
- [ ] Full training (shared 2ep + count 3ep)
- [ ] Per-type EDA (24개 유형별 accuracy)
- [ ] 첫 3-5회 leaderboard 제출

### 점수 상승폭 큰 것 (Day 2-3)
- [ ] LoRA r=64 A/B test
- [ ] Cosine scheduler A/B test
- [ ] Dev confidence gradient weight 최적화
- [ ] 24-type router + Top 5 특화 prompt
- [ ] Ensemble weight optimization

### 시간 남으면 할 것 (Day 3-4)
- [ ] Full-text logprob (mean-normalized)
- [ ] 2-model ensemble (seed=42 + seed=123)
- [ ] 불확실 샘플 2차 고해상도 추론
- [ ] AI Hub material synthetic (EDA 결과에 따라)
- [ ] Material/Object expert adapter (3-adapter)

---

## 13. 최종 추천

**시간이 극히 부족하면:**
- Qwen3.5-9B (또는 smoke test 승자) + BF16 + 기존 2-adapter + logprob reranking
- 이것만으로도 baseline 대비 **+5-10pp** 기대

**1등 ceiling을 노리면:**
- Qwen3.5-9B BF16 shared + count expert
- 24-type router + Top 5 특화 prompt
- Dev confidence gradient + EDA-driven targeted 강화
- Option permutation averaging (3/5)
- Full-text logprob (mean-normalized)
- Ensemble weight validation 최적화
- 2-model ensemble (다른 seed)

핵심은 한 문장:

> **"먼저 돌려서 어디가 약한지 보고, 약한 곳만 정밀하게 치료하라."**

---

## 14. 실전 환경

| 항목 | 설정 |
|---|---|
| **Hardware** | Colab H100 (80GB VRAM) |
| **Data** | Google Drive 마운트 (`/content/drive/MyDrive/`) |
| **Notebook** | `ssafy_vqa_qwen3vl_h100_topscore.ipynb` |
| **제출** | 하루 최대 20회 |
| **Deadline** | 2026-04-06 |
| **flash-attn** | 삭제 (SDPA 사용) |
| **HF 다운로드** | 삭제 (Drive 마운트) |
