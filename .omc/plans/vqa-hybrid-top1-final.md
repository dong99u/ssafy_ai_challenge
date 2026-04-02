# VQA Competition HYBRID TOP-1 Plan -- H100 80GB FINAL

**Created:** 2026-04-02
**Revised:** 2026-04-02 (v8 -- Claude x ChatGPT Hybrid, deliberate mode)
**Deadline:** 2026-04-06 (4 days remaining)
**Hardware:** Colab H100 (80GB VRAM)
**Data:** Google Drive mount
**Base:** `ssafy_vqa_qwen3vl_h100_topscore.ipynb`
**Objective:** TOP-1 accuracy on SSAFY 15th AI Challenge
**Origin:** Claude 실행 프레임 + ChatGPT 모델링 코어 하이브리드

---

## RALPLAN-DR Summary

### Principles (5)

1. **Claude를 운영체제로, ChatGPT를 모델링 코어로**: 실행 프레임워크(A/B gate, 안전 제출, 리스크 관리)는 Claude안, 데이터/모델 설계(expert 구조, 외부데이터, q_type 앙상블)는 ChatGPT안을 따른다.
2. **H100 = BF16 full precision**: 4-bit QLoRA는 로컬/빠른 ablation용만. H100 메인 학습은 반드시 BF16. (합의: 양쪽 모두 동의)
3. **Expert 분업은 선택이 아닌 기본 구조**: shared + material/object expert + count expert + OCR 보조. 단일 모델만 쓰면 ceiling 한계. (ChatGPT안 채택)
4. **외부데이터는 조건부가 아니라 우선순위**: AI Hub material/object synthetic → TACO count synthetic → TrashNet material 보조 순서로 즉시 투입. (ChatGPT안 채택, Claude의 "조건부" 기각)
5. **A/B gate + Safe submission 우선**: 모든 변경은 >= 0.5pp 향상 시에만 채택. Day 2 저녁까지 반드시 첫 제출. (Claude안 유지)

### Decision Drivers (Top 3)

1. **material_general + object_type가 test의 ~40.7%** (21.9%+18.8%): 이 두 카테고리를 전문가 모델로 강화하는 것이 점수 상승의 최대 레버. train/test 분포가 일치하지만 dev는 count-heavy라 다름.
2. **H100 80GB = BF16 + batch 4-8 + 20회/일 제출**: quantization 불필요. 하루 6-8 full run. Leaderboard 기반 실시간 A/B 검증 가능.
3. **텍스트 편향이 강하다**: material/object에서 텍스트만으로 50%+ 정확도. text prior + template-aware CV가 필수. count만 vision 의존도 높음.

### Viable Options (3)

#### Option A: Conservative Single-Model (REJECTED)
- shared + count 2-adapter만 유지
- 외부데이터 조건부
- **Expected gain:** +3-6pp
- **Why rejected:** material/object가 test의 52%인데 전문가 없음. Ceiling 한계.

#### Option B: Hybrid Expert System (CHOSEN)
- shared + material/object expert + count expert + OCR 보조
- AI Hub + TACO + TrashNet 즉시 투입
- Claude 실행 프레임 (A/B gate, smoke test, safe submission)
- template-aware CV + per-qtype weight optimization
- **Expected gain:** +6-15pp
- **Risk:** MEDIUM -- expert 학습 시간 추가. 하지만 H100 속도로 흡수 가능.
- **Time:** 4 days (Day 4 = final ensemble + safety)

#### Option C: Full Data-Heavy (REJECTED)
- Expert 구조 + 대규모 외부데이터 (AI Hub 전체 20GB+)
- **Why rejected:** 데이터 pipeline 코딩에 ~1일 소요. 시간 ROI 낮음. 핵심 카테고리만 타겟팅이 효율적.

### 두 전략 간 핵심 합의/비합의 매트릭스

| 항목 | Claude안 | ChatGPT안 | **합본 결정** | 이유 |
|------|---------|-----------|-------------|------|
| H100 학습 방식 | BF16 full precision | 4-bit QLoRA | **BF16** (Claude) | H100에서 4-bit는 ceiling 제한 |
| 모델 구조 | shared + count (2-adapter) | shared + material/object + count + OCR | **4-expert** (ChatGPT) | material/object가 test 52% |
| 외부데이터 | 조건부 (EDA 후) | 즉시 우선순위 | **즉시 투입** (ChatGPT) | 1등 ceiling 추구 |
| Dev 활용 | confidence-based weighting | count expert에만 강하게 | **count 강하게 + confidence 차등** (합성) | 분포 인식 + 신뢰도 |
| Scoring | full-text mean-logprob A/B | letter reranking + q_type별 결합 | **q_type별 구조 + mean-logprob A/B** (합성) | 구조는 ChatGPT, 구현은 Claude |
| CV/Validation | per-qtype weight optimization | template-aware group split | **template-aware fold + per-qtype opt** (합성) | 과낙관 방지 |
| 실행 프레임 | A/B gate, smoke test, safe submit | 없음 (전략만) | **Claude 프레임** 전체 채택 | 실행 안전성 |
| Text prior | 있으나 약함 | 강하게 권장 (LightGBM/TF-IDF) | **강하게** (ChatGPT) | 텍스트 편향 50%+ |

---

## PRE-MORTEM (--deliberate: 4 failure scenarios)

### Scenario 1: Material/Object Expert가 shared보다 못함
**확률:** 15%
**원인:** AI Hub synthetic VQA의 도메인 갭이 크거나, material/object 질문이 실제로는 이미지 의존도가 낮아 shared+text prior로 충분
**탐지:** Day 2 A/B에서 material expert <= shared - 0.5pp
**대응:** material expert 폐기, shared에 AI Hub 데이터를 약하게 섞음. text prior 가중치 상향. 이미 safe submission 있으므로 손실 최소.

### Scenario 2: Qwen3.5-9B가 한국어 VQA에서 기대 이하
**확률:** 25%
**원인:** Qwen3.5-9B의 한국어 fine-tuning 데이터가 VL 태스크에서 약함. 벤치마크는 영어 중심.
**탐지:** Day 1 smoke test에서 Qwen3.5-9B < Qwen3-VL-8B - 2pp
**대응:** Qwen3-VL-8B로 즉시 전환. Qwen2.5-VL-7B를 2차 fallback. 코드 변경 최소 (AutoModel 사용).

### Scenario 3: 시간 부족으로 expert 구조 미완성
**확률:** 20%
**원인:** Day 1-2에서 BF16 전환/모델 선정에 예상보다 오래 걸림. AI Hub 데이터 전처리 지연.
**탐지:** Day 2 저녁까지 material expert 학습 미시작
**대응:** expert 구조를 포기하고 "shared + count + text prior" 3-channel로 축소. 이것만으로도 baseline 대비 +4-8pp 기대. Day 3-4는 ensemble weight optimization에 집중.

### Scenario 4: H100 할당량 소진 / 런타임 불가 (Architect mandatory)
**확률:** 10-15%
**원인:** Colab H100 GPU quota 소진. 또는 H100 인스턴스 자체가 4일간 안정적으로 유지되지 않음.
**탐지:** Colab에서 H100 런타임 연결 실패. 또는 연속 3회 이상 세션 끊김.
**대응:** 즉시 T4 또는 A100 런타임으로 전환.
  - `CFG.train_use_4bit = True` 복원 (4-bit QLoRA 모드)
  - batch size 1-2로 축소
  - resolution 1280x1280으로 하향
  - material expert 포기 → shared + count 2-adapter로 축소
  - 기존 노트북의 QLoRA 설정이 이미 동작하므로 코드 변경 최소
  - 이 degraded mode에서도 baseline 대비 +2-4pp 가능 (epoch 증가 + permutation)

---

## EXPANDED TEST PLAN (--deliberate)

### Unit Tests (코드 정합성)
- [ ] `score_fulltext_with_logprob()`: 단일 샘플에서 length normalization 확인 (짧은/긴 선택지 score 비교)
- [ ] `classify_question_detail()`: 24개 유형 분류기 정확도 >= 95% (train set 기준)
- [ ] `build_material_synthetic_vqa()`: AI Hub 데이터 → VQA 포맷 변환 정합성 (질문/선택지/정답 매칭)
- [ ] `template_aware_group_split()`: 동일 템플릿이 같은 fold에 안 들어가는지 검증
- [ ] `text_prior_model()`: TF-IDF + LightGBM이 material 질문에서 random보다 유의하게 높은지

### Integration Tests (파이프라인 연결)
- [ ] BF16 model load → LoRA attach → 1 step forward/backward → OOM 없음 (H100)
- [ ] Expert routing: q_type → 올바른 adapter 선택 → 올바른 blend weight 적용
- [ ] `combine_probs` 리팩토링 후: 5-channel 입력 (shared/count/text/ocr/material) → 정규화된 확률 출력. material_probs=0일 때 기존 4-channel 결과와 동일한지 regression test
- [ ] 3-adapter inference: `PeftModel.load_adapter()` 3회 → material/shared/count 각각 정상 로드 + 메모리 충돌 없음
- [ ] Dev pseudo label → confidence filter → count expert training data로 정상 합류
- [ ] Full inference pipeline: image load → q_type classify → expert select → logprob score → ensemble → 최종 답

### E2E Tests (최종 품질)
- [ ] Day 1 baseline vs v5 plan baseline: BF16 전환만으로 >= 0pp (regression 없음)
- [ ] Day 2 safe submission: leaderboard score 기록. 내부 validation과 ±3pp 이내 일치.
- [ ] Per-qtype accuracy breakdown: material/object/count 각각 기록. 약점 유형 Top 3 식별.
- [ ] Final ensemble vs single best model: >= 0.5pp 향상 확인

### Observability
- [ ] 매 training run: loss curve + validation accuracy + VRAM peak 로그
- [ ] 매 submission: leaderboard score + 내부 validation score + 제출 config snapshot
- [ ] Per-qtype dashboard: 24개 유형별 accuracy 변화 추적 (Day 1 → Day 4)
- [ ] Expert 기여도: ensemble에서 각 expert의 실제 기여 비율 기록

---

## DELTA PLAN: 7 Targeted Improvements (Hybrid)

### ROI Assessment

| Delta | Expected Gain | Confidence | Risk | Time Cost | Source |
|-------|--------------|------------|------|-----------|--------|
| Delta 1: BF16 + Batch + Resolution | +1-3pp | HIGH | LOW | ~1h | Claude |
| Delta 2: Model Selection (3-way smoke) | +3-8pp | MEDIUM-HIGH | MEDIUM | ~2h | Claude |
| Delta 3: Training Intensity (Epoch/Scheduler/LoRA/Perm) | +1-2pp | MEDIUM-HIGH | LOW | ~3h | Claude |
| **Delta 4: Material/Object Expert + AI Hub Synthetic** | **+2-5pp** | **MEDIUM** | **MEDIUM** | **~4h** | **ChatGPT** |
| **Delta 5: Count Expert + Dev Soft Label + TACO** | **+1-3pp** | **MEDIUM** | **LOW** | **~3h** | **ChatGPT** |
| Delta 6: Full-text Logprob + Text Prior | +1-3pp | MEDIUM | LOW | ~4h | 합성 |
| Delta 7: Ensemble Optimization + Template-aware CV | +0.5-2pp | MEDIUM-LOW | LOW | ~2h | 합성 |
| **Combined realistic gain** | **+8-12pp** | | | *gains are not fully additive* |

---

### Delta 1: H100 Full Utilization (HIGHEST PRIORITY -- Day 1 오전)

**출처:** Claude안 그대로 유지

```python
CFG.train_use_4bit = False          # BF16 full precision
CFG.per_device_batch_size = 4       # was: 1
CFG.grad_accum_steps = 4            # was: 8 (effective=16)
CFG.train_min_pixels = 512 * 512    # was: 448*448
CFG.train_max_pixels = 1536 * 1536  # was: 1280*1280
CFG.infer_min_pixels = 672 * 672
CFG.infer_max_pixels = 1792 * 1792
CFG.infer_ocr_max_pixels = 2048 * 2048
```

**Acceptance:** BF16 load 성공, batch 4+ training, peak VRAM <= 55GB

---

### Delta 2: 3-Model Smoke Test (HIGHEST PRIORITY -- Day 1 오전)

**출처:** Claude안 유지 (ChatGPT도 동일 모델 순위)

- Qwen3.5-9B vs Qwen3-VL-8B vs Qwen2.5-VL-7B
- 각 1-epoch BF16 smoke test (~40min)
- Winner = PRIMARY MODEL
- `AutoModelForImageTextToText` 사용 (모든 모델 호환)
- SDPA only (flash-attn 설치 시간 절약)

**⚠️ Qwen3.5-9B 사전 검증 (Critic flag):**
- smoke test 전 반드시 확인: `Qwen/Qwen3.5-9B`가 vision input을 지원하는 VL 모델인지?
- 검증 방법: `model.config` 출력 → `vision_config` 존재 여부 확인
- 또는 단일 이미지 forward pass → 성공/실패로 즉시 판정 (~5min)
- 실패 시: Qwen3.5-9B 포기 → Qwen3-VL-8B vs Qwen2.5-VL-7B 2-way 비교로 축소

**Go/No-Go:** Winner accuracy >= 2nd place - 1pp → GO. 그 외 → 2nd place 채택.

---

### Delta 3: Training Intensity (HIGH PRIORITY -- Day 1 오후~Day 2)

**출처:** Claude안 유지

```python
CFG.shared_epochs = 2   # was: 1
CFG.count_epochs = 3    # was: 2
CFG.n_perm_shared = 3   # was: 2
CFG.n_perm_count = 5    # was: 3
# Cosine scheduler
# LoRA r=64, alpha=128 A/B test
# BF16에서 LR = 5e-5 ~ 1e-4 (QLoRA용 2e-4는 발산 위험)
```

---

### Delta 4: Material/Object Expert + AI Hub Synthetic (KEY DIFFERENTIATOR -- Day 2)

**출처:** ChatGPT안 핵심. Claude에서 "조건부"였던 것을 "즉시"로 격상.

**근거:**
- material_general: train 23.1%, test 21.9% → 최대 단일 카테고리
- object_type: train 16.7%, test 18.8% → 2위 카테고리
- 합계 ~40%가 material/object. 전문가 없이는 ceiling 한계.

**4a: AI Hub Synthetic VQA 생성**

**⚠️ AI Hub 데이터 소스 명시:**
- "AI Hub 생활 폐기물 이미지" (aihub.or.kr, 데이터셋 ID 확인 필요)
- "AI Hub 재활용품 분류 및 선별 데이터" (aihub.or.kr)
- 다운로드 방법: aihub API 또는 수동 다운로드 후 Google Drive 업로드
- 어노테이션 형식: JSON (카테고리 라벨 + bbox) → VQA 변환 필요

```python
# AI Hub 데이터 → 4지선다 VQA 변환
# 예시:
# Q: 사진 속 주요 재활용품의 재질은 무엇인가요?
# A: 유리 / 금속 / 종이 / 플라스틱 → 정답: 플라스틱
#
# Q: 사진에 보이는 재활용 가능한 물체의 종류는 무엇인가요?
# A: 캔 / 유리병 / 플라스틱 병 / 종이팩 → 정답: 플라스틱 병

def build_material_synthetic_vqa(aihub_data, templates):
    """AI Hub 데이터를 대회 포맷과 동일한 4지선다 VQA로 변환"""
    # material_general 템플릿 5-10종
    # object_type 템플릿 5-10종
    # recycle_class 템플릿 3-5종
    # Hard negative mining: 같은 대분류 내 다른 소분류를 오답으로
    pass
```

**4a-gate: AI Hub 데이터 품질 게이트 (Architect mandatory)**

Synthetic VQA 학습 전 반드시 수행:
1. 생성된 synthetic VQA에서 50개 샘플 무작위 추출
2. 수동 검증 (30분):
   - [ ] 질문 템플릿이 대회 형식과 일치하는가?
   - [ ] 이미지 도메인이 대회 이미지와 시각적으로 유사한가? (조명, 구도, 배경)
   - [ ] 선택지 분포가 균형적인가? (특정 답에 편중되지 않는가?)
   - [ ] 정답 라벨이 올바른가? (annotation → VQA 변환 과정에서 오류 없는가?)
3. 50개 중 >= 45개 통과 시 → GO (학습 진행)
4. 40-44개 통과 → 문제 유형 수정 후 재검증
5. < 40개 통과 → AI Hub synthetic 포기, TrashNet만 사용

**4b: Material/Object Expert Adapter**

**학습 데이터 구성 (Architect recommended — full data + upsampling):**
- 전체 train 데이터를 사용하되, material/object 샘플을 **3x upsampling**
- ❌ material/object 샘플만으로 exclusive 학습 (데이터 단편화 위험)
- ✅ 전체 데이터 + material/object 3x → 일반 능력 보존 + 전문화
- AI Hub synthetic도 weight 0.3-0.5로 추가 (높은 weight 금지 — 도메인 갭)

```python
# 기존 shared + count 2-adapter → shared + material_object + count 3-adapter
# blend weight 초기값 (ChatGPT안, validation opt로 fine-tune):
#   material: 0.45*material_expert + 0.30*shared + 0.20*text_prior + 0.05*OCR
#   object:   0.45*material_expert + 0.30*shared + 0.20*text_prior + 0.05*OCR
```

**4c: combine_probs 리팩토링 (Critic mandatory — ~1h)**

기존 `combine_probs` 함수는 `(shared_probs, count_probs, text_probs, ocr_probs)` 4개 입력만 지원.
3-adapter 시스템을 위해 `material_probs` 파라미터 추가 필요.

```python
# BEFORE (현재 코드 line ~1346-1358):
# def combine_probs(shared_probs, count_probs, text_probs, ocr_probs, weights):
#     final = (weights["shared"]*shared_probs + weights["count"]*count_probs
#              + weights["text"]*text_probs + weights["ocr"]*ocr_probs)
#     final = final / final.sum()  # ← 이미 정규화됨 (Architect note)
#     return final

# AFTER:
# def combine_probs(shared_probs, count_probs, text_probs, ocr_probs,
#                   material_probs, weights):
#     final = (weights.get("shared",0)*shared_probs
#              + weights.get("count",0)*count_probs
#              + weights.get("text",0)*text_probs
#              + weights.get("ocr",0)*ocr_probs
#              + weights.get("material",0)*material_probs)
#     final = final / final.sum()  # 기존 정규화 유지 — weights 합 != 1.0이어도 OK
#     return final

# ensemble_weights 업데이트:
# "material": {"shared": 0.30, "count": 0.00, "text": 0.20, "ocr": 0.05, "material": 0.45},
# "object":   {"shared": 0.30, "count": 0.00, "text": 0.20, "ocr": 0.05, "material": 0.45},
# (기존 키: count/color/location/ocr_text 등은 "material": 0.00 추가)
```

**추론 라우팅:** 기존 inference loop에서 material expert adapter를 `PeftModel.load_adapter()`로 추가 로드.
q_type에 따라 올바른 adapter 선택 → logprob scoring → combine_probs로 앙상블.

**4d: Delta 4 롤백 절차:**
- AI Hub synthetic이 오염된 경우: material expert checkpoint 삭제 → shared+count 2-adapter로 복귀
- 복귀 시 사용할 checkpoint: Day 1 저녁 `SUBMISSION #1-3` 시점의 shared+count adapter
- 롤백 소요 시간: ~30min (checkpoint 교체만)

**TrashNet 보조:** glass/metal/paper/plastic 분류 → material 학습 보조 데이터

**A/B Test:** shared-only vs shared+material_expert (material/object qtype에서, template-aware CV 기준)
**Acceptance:** material/object accuracy에서 >= 1pp 향상

---

### Delta 5: Count Expert + Dev Soft Label + TACO (HIGH PRIORITY -- Day 2~3)

**출처:** ChatGPT안의 dev 활용 + Claude안의 confidence weighting 합성

**5a: Dev 활용 원칙 (합성안)**
- dev는 count-heavy (71.4%) → **count expert에만 강하게** 투입 (ChatGPT안)
- shared model에는 약하게 또는 제외 (ChatGPT안)
- 내부 weighting은 confidence 차등 (Claude안):
  ```python
  # 5/5 일치 → weight 0.95
  # 4/5 일치 → weight 0.70
  # 3/5 일치 (count만) → weight 0.40
  # 2-2-1 이하 → 버림
  ```

**5b: TACO Count Synthetic VQA**
```python
# TACO bbox → count 질문 생성
# Q: 사진에 보이는 플라스틱 병은 몇 개입니까?
# A: 1개 / 2개 / 3개 / 4개 → 정답: 3개
```

**5c: Soft Label Training (stretch)**
- 5명 annotator 분포를 soft distribution으로 → KL divergence loss
- 불확실한 샘플에서 더 robust한 학습

---

### Delta 6: Full-text Logprob + Text Prior (MEDIUM PRIORITY -- Day 3)

**출처:** 합성 -- Claude의 logprob 구현 + ChatGPT의 text prior 구조

**6a: Full-text Mean-logprob Scoring (Claude안)**

```python
def score_fulltext_with_logprob(
    model, processor, image, prefix_messages,
    choice_texts: List[str],  # ["유리", "금속", "종이", "플라스틱"]
    normalize: bool = True,    # mean-log-prob (길이 정규화)
) -> np.ndarray:
    """score = (1/n_tokens) * sum_t log P(token_t | ...)"""
    pass

# A/B: letter-only vs full-text vs ensemble
```

**6b: Text Prior Branch (ChatGPT안)**

```python
# 입력: question + [SEP] + a/b/c/d 선택지 텍스트
# 모델: char/word TF-IDF + LightGBM (빠르고 가벼움)
# 출력: option prior score (4-dim)
#
# 결합 가중치 (ChatGPT안):
#   material/object/recycle/location: text prior 높게 (0.20-0.25)
#   count: text prior 낮게 (0.10)
#
# 근거: material naive text-only acc 56.5%, count는 26.2%
```

---

### Delta 7: Ensemble Optimization + Template-aware CV (LOW PRIORITY -- Day 3~4)

**출처:** 합성 -- Claude의 validation optimization + ChatGPT의 template-aware CV

**7a: Template-aware Group Split (ChatGPT안)**

```python
def template_aware_group_split(df, n_splits=5):
    """
    질문 템플릿을 정규화하여 그룹화.
    같은 템플릿 그룹은 같은 fold에 들어가도록.
    일반 KFold는 템플릿 반복으로 과낙관.
    """
    # "사진에 보이는 재활용 가능한 플라스틱 병은 몇 개입니까?"
    # "사진 속 재활용품 중 플라스틱 병은 몇 개 있나요?"
    # → 둘 다 "count_bottle" 템플릿 그룹
    pass
```

**7b: Per-qtype Ensemble Weight Optimization (Claude안)**

```python
# scipy.optimize 또는 grid search
# template-aware fold 위에서 per-qtype weight 탐색
from scipy.optimize import minimize
# Regularization으로 validation overfitting 방지
```

**7c: Q_type별 최종 앙상블 구조 (ChatGPT안)**

| q_type | material_expert | count_expert | shared | text_prior | OCR |
|--------|----------------|-------------|--------|-----------|-----|
| material/object/recycle | 0.45 | 0.00 | 0.30 | 0.20 | 0.05 |
| count | 0.00 | 0.45 | 0.25 | 0.10 | 0.20(detector) |
| color | 0.25 | 0.00 | 0.45 | 0.20 | 0.10 |
| location/brand/other | 0.00 | 0.00 | 0.35 | 0.25 | 0.25+0.15(fallback) |

→ 이 초기값에서 validation optimization으로 fine-tune

---

## TIMELINE (4 Days -- Hybrid)

| Day | Block | Activity | Deliverable |
|-----|-------|----------|-------------|
| **Day 1** | 오전 | **Delta 1+2:** BF16 전환 + 3-model smoke test (Qwen3.5-9B/Qwen3-VL-8B/Qwen2.5-VL-7B) | **모델 결정 확정** |
| | 오후 | **Delta 3 시작:** 선택 모델로 full training (shared 2ep + count 3ep) | Full-trained baseline |
| | 저녁 | Validation + **SUBMISSION #1-3**. AI Hub 데이터 다운로드 시작 (백그라운드) | 첫 leaderboard + 외부데이터 준비 |
| **Day 2** | 오전 | **Delta 3:** A/B tests (LoRA r, scheduler, epochs). **Delta 4a:** AI Hub → synthetic VQA 변환 | A/B 결과 + synthetic 데이터 |
| | 점심 | **Delta 4a-gate:** AI Hub 품질 게이트 (50샘플 검증, 30min). GO/NO-GO 결정 | 품질 게이트 통과 여부 |
| | 오후 | **Delta 4b-c:** Material expert 학습 + `combine_probs` 리팩토링 (~1h 추가) | Material expert + 라우팅 코드 |
| | 저녁 | **SUBMISSION #4-8** (A/B winners + material expert). | Expert 검증 |
| **Day 2→3** | 야간 | **Delta 5 시작:** Dev soft label 준비 + TACO synthetic. Text prior 학습 시작 | Count 데이터 + text prior |
| **Day 3** | 오전 | **Delta 5:** Count expert 학습 (dev + TACO). **Delta 6:** Full-text logprob 구현 | Count expert + logprob |
| | 오후 | **Delta 6:** Text prior A/B. **Delta 7:** Template-aware CV + ensemble weight opt | Logprob + ensemble 최적화 |
| | 저녁 | **SUBMISSION #9-14**. Best config 확정. 24-type router 구현 | Best config |
| **Day 4** | 오전 | Best config retrain (더 많은 epochs). 2nd model for ensemble (seed=123) | Final models |
| | 오후 | 2-model ensemble + option permutation averaging. Uncertain-only 2차 추론 | Ensemble submission |
| | 저녁 | **FINAL SUBMISSION #15-20**. Safety backup 30min 전 | Final |

### Submission Budget (20회/일)

| Day | 예산 | 용도 |
|-----|------|------|
| Day 1 | 3-4회 | Baseline variants + leaderboard calibration |
| Day 2 | 5-8회 | A/B winners + material expert + count expert |
| Day 3 | 5-7회 | Logprob + text prior + ensemble variants |
| Day 4 | 5-8회 | Final ensemble + safety backups |

---

## RISK MITIGATION

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Material expert가 shared보다 못함 | 15% | MEDIUM | A/B gate. 폐기 후 shared + text prior로 전환 |
| Qwen3.5-9B 한국어 약함 | 25% | MEDIUM | Day 1 smoke test. Qwen3-VL-8B fallback |
| AI Hub 데이터 노이즈 | 20% | LOW | Synthetic VQA 품질 검사 + weight 0.3-0.5로 제한 |
| 시간 부족 (expert 미완성) | 20% | MEDIUM | shared+count+text_prior 3-channel 축소 모드 |
| Colab session 끊김 | MEDIUM | HIGH | 200 steps마다 checkpoint → Google Drive sync |
| **H100 할당량 소진** | **10-15%** | **HIGH** | **T4/A100 fallback: `train_use_4bit=True`, batch 1-2, 1280px. shared+count 2-adapter 축소** |
| Template-aware CV 과소적합 | 10% | LOW | 일반 StratifiedKFold와 비교. 더 나은 쪽 채택 |
| 3-adapter checkpoint 저장공간 | LOW | LOW | Google Drive 15GB 무료. adapter당 ~200MB x 3 x A/B runs ≈ ~3GB. 충분. 오래된 checkpoint 정리 |
| Qwen3.5-9B가 VL 모델이 아닌 경우 | LOW | CRITICAL | Day 1 smoke test에서 이미지 입력 가능 여부 즉시 확인. 실패 시 Qwen3-VL-8B로 전환 |

---

## FALLBACK CHAIN

```
Full Expert (4-expert + 외부데이터 + text prior + template CV)
  ↓ 시간 부족 or expert 실패
Reduced Expert (shared + count + text prior, 외부데이터 일부)
  ↓ 심각한 시간 부족
Conservative (shared only + BF16 + batch 4 + resolution up)
  ↓ 모든 것 실패
v5 Baseline (기존 노트북 그대로 H100에서 실행)
```

---

## ADR (Architecture Decision Record)

**Decision:** Claude 실행 프레임 위에 ChatGPT 모델링 코어를 결합한 하이브리드 전략으로 4일간 실행.

**Drivers:**
1. test에서 material_general(21.9%) + object_type(18.8%) = 40.7%가 핵심. 전문가 없이는 ceiling 한계.
2. dev가 count-heavy(71.4%)라 shared에 강하게 섞으면 분포 오염. count expert에만 투입이 안전.
3. 텍스트 편향이 강해서 (material naive 56.5%) text prior가 무료 점수원.
4. H100 80GB에서 BF16이 4-bit보다 확실히 우월 (양쪽 합의).
5. 20회/일 제출로 leaderboard A/B testing 가능.

**Alternatives considered:**
- **Claude안 단독:** 실행 안전하나 material/object expert 없이 ceiling 제한. Rejected.
- **ChatGPT안 단독:** 모델링 깊지만 실행 프레임(A/B gate, 안전 제출) 부재. Rejected.
- **Option C Data-Heavy:** AI Hub 전체 다운로드는 시간 ROI 낮음. Rejected.

**Why chosen:** "Claude로 어떻게 안전하게 빠르게" + "ChatGPT로 무엇을 맞춰야 크게 오를지" 합성이 최적.

**Consequences:**
- Day 1에 모델 결정 + BF16 baseline 확립
- Day 2에 expert 구조 검증 (material/object + count)
- Day 3에 scoring/ensemble 최적화
- Day 4에 final ensemble + safety

**Follow-ups:**
- Per-qtype accuracy tracking으로 약점 실시간 식별
- Leaderboard score와 내부 validation 괴리 모니터링
- Expert 기여도 분석 → 불필요한 expert 제거
