# SSAFY 15기 AI Challenge — 최종 전략 정리

## 대회 개요

| 항목 | 내용 |
|---|---|
| **과제** | 재활용품 이미지 VQA (Visual Question Answering) |
| **입력** | 재활용품 이미지 + 한국어 4지선다 질문 (a/b/c/d) |
| **출력** | `id,answer` CSV (answer는 a/b/c/d 중 하나) |
| **평가지표** | Accuracy |
| **데이터** | train 5,073 / test 5,074 / dev 4,413 (5명 어노테이터 응답) |

---

## 1. 모델 선택

| 항목 | 베이스라인 | 최종 |
|---|---|---|
| **모델** | Qwen2.5-VL-**3B**-Instruct | Qwen2.5-VL-**7B**-Instruct |
| **양자화** | 4-bit NF4 (BitsAndBytes) | 4-bit NF4 (BitsAndBytes) |
| **GPU** | RTX 5060 Ti | **H100 80GB** |

**선택 근거**: H100 80GB VRAM으로 7B 모델을 4-bit 양자화 상태에서 안정적으로 학습 가능. 3B → 7B 스케일업으로 이미지 이해력 및 한국어 처리 능력 향상.

---

## 2. Fine-tuning 전략 (LoRA + QLoRA)

### 하이퍼파라미터

| 파라미터 | 베이스라인 (3B) | Model A (7B) | Model B (7B) |
|---|---|---|---|
| **LoRA rank (r)** | 8 | 64 | 64 |
| **LoRA alpha** | 16 | 128 | 128 |
| **LoRA dropout** | — | 0.05 | 0.05 |
| **Learning Rate** | 1e-4 | **1e-5** | **5e-5** |
| **Epochs** | 1 | 3 | 3 |
| **Scheduler** | linear | **cosine** | **cosine** |
| **Warmup ratio** | 0.03 | 0.05 | 0.05 |
| **Batch size** | 1 | 1 | 1 |
| **Grad accumulation** | 4 | 4 | 4 |
| **Seed** | — | 42 | 123 |
| **이미지 해상도** | 384px | **672px** | **672px** |

### 변경 근거

- **LoRA rank 8→64**: fine-tuning 용량 8배 증가. 재활용품 분류 같은 도메인 특화 작업에 더 많은 파라미터 적응 필요.
- **LR 1e-4→1e-5**: LoRA rank 증가에 따른 보상. 큰 rank에서 높은 LR은 불안정하므로 10배 감소.
- **Epochs 1→3**: 1 epoch은 underfitting 우려. 3 epoch + cosine scheduler로 후반부 수렴 개선.
- **이미지 672px**: 재활용품 세부 특징(라벨, 재질, 광택 등) 인식에 고해상도 필수. H100 VRAM으로 가능.
- **alpha/rank 비율 1:2 유지**: LoRA 논문 권장 비율 준수 (alpha = 2 × rank).

### LoRA 타겟 모듈

Attention + MLP projection 레이어에 적용 (PEFT 기본 설정).

---

## 3. 데이터 전략

### FULL_DATA_MODE (전체 데이터 학습)

최종 제출용으로 **train + dev 전체 데이터로 학습**:

- `train.csv` (5,073개) + `dev.csv` (4,413개, majority vote로 단일 answer 변환)
- 검증셋은 학습 데이터에서 100개 샘플 (모니터링용)
- Early stopping 사실상 비활성화 (patience=999)
- 고정 3 epoch 학습

### dev.csv 활용

- 5명 어노테이터 응답(`answer1`~`answer5`)에서 **다수결 투표**로 gold label 생성
- 학습 데이터 약 **2배 확보** (5,073 → 9,486)

---

## 4. 프롬프트 엔지니어링

### System Prompt (재활용 분류 전문가)

```
당신은 재활용 분류 전문가입니다. 재활용품 이미지를 보고 질문에 정확히 답변합니다.

핵심 지침:
1. 개수를 세는 질문: 이미지를 꼼꼼히 살펴보고 해당 물품의 정확한 개수를 세세요.
   겹치거나 부분적으로 보이는 물품도 포함합니다.
2. 재질 판별: 플라스틱(투명/불투명, 유연/경질), 유리(투명/색유리),
   금속(알루미늄/철), 종이(골판지/일반), 비닐/스티로폼을 구분하세요.
3. 금속 vs 플라스틱: 광택, 반사, 찌그러짐 패턴으로 구분합니다.
   금속은 광택이 있고 찌그러지면 주름이 생깁니다.
   플라스틱은 무광이며 라벨이 붙어있는 경우가 많습니다.

반드시 a, b, c, d 중 하나의 소문자 한 글자로만 답하세요.
```

### User Prompt (객관식 포맷)

```
{질문}
(a) {선택지a}  (b) {선택지b}  (c) {선택지c}  (d) {선택지d}

정답을 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요.
```

**설계 의도**: 재활용 도메인에 특화된 지침(개수 세기, 재질 판별, 금속/플라스틱 구분)을 제공하여 모델이 이미지를 더 세심하게 분석하도록 유도.

---

## 5. 추론 전략

### 5-1. Logprob 기반 추론 (generate 대신)

`model.generate()` 대신 **forward pass로 logprob 직접 추출**:

- a, b, c, d 각 토큰의 log-probability를 계산
- 가장 높은 logprob의 선택지를 정답으로 선택
- 장점: 더 정확한 확률 비교, 앙상블/TTA에 활용 가능

### 5-2. TTA (Test-Time Augmentation) — 선택지 셔플링

**Position bias 제거**를 위한 4가지 순서 조합:

| Permutation | 순서 |
|---|---|
| Original | a, b, c, d |
| Shuffle 1 | b, d, a, c |
| Shuffle 2 | c, a, d, b |
| Shuffle 3 | d, c, b, a |

각 permutation에서:
1. 선택지를 셔플된 순서로 재배치
2. 모델에 입력하여 logprob 추출
3. logprob을 **원래 선택지 위치로 역매핑**
4. 4개 permutation의 logprob을 **평균**하여 최종 예측

**효과**: VLM이 특정 위치(예: 첫 번째 선택지)를 선호하는 position bias를 제거.

### 5-3. 앙상블 (Model A + Model B)

- Model A (seed=42, lr=1e-5) + Model B (seed=123, lr=5e-5)
- **Logprob 평균 앙상블**: 두 모델의 a/b/c/d logprob을 평균 후 argmax
- 자동 선택: val accuracy 기준으로 TTA vs 앙상블 중 우수한 쪽을 최종 제출

---

## 6. 실험 결과

| # | 실험 | Public Accuracy | 비고 |
|---|---|---|---|
| 1 | 베이스라인 (3B, 5060 Ti) | 미측정 | 로컬 테스트용 |
| 2 | 7B + TTA (튜닝 전) | **0.91525** | 하이퍼파라미터 튜닝 전 기준선 |
| 3 | 7B + 튜닝 + TTA (Model A) | **0.91761** | +0.00236 향상 |
| 4 | 7B + Model B 단독 | 미측정 | 앙상블용 |
| 5 | 7B + 앙상블 (A+B) | 0.91328 | TTA보다 하락 (-0.00433) |

### 최고 성능: **0.91761** (Model A + TTA)

### 앙상블 하락 원인 분석

- FULL_DATA_MODE에서 validation이 학습 데이터 일부(100샘플)라 앙상블 우위가 **과적합된 수치**
- 같은 데이터로 학습한 두 모델의 앙상블은 **다양성 부족**
- 교훈: 앙상블은 서로 다른 데이터 분할 또는 다른 모델 아키텍처일 때 효과적

---

## 7. 제출 파이프라인

```
submission_model_a.csv   — Model A 단독 (no TTA)
submission_tta.csv       — Model A + TTA (4 permutations)  ← 최종 제출
submission_ensemble.csv  — Model A + B logprob 앙상블
submission.csv           — 자동 선택 (val 기준 best)
```

---

## 8. 핵심 기술 스택

| 카테고리 | 기술 |
|---|---|
| **Base Model** | Qwen/Qwen2.5-VL-7B-Instruct |
| **Quantization** | 4-bit NF4 (BitsAndBytes) |
| **Fine-tuning** | LoRA (PEFT), r=64, alpha=128 |
| **Training** | AdamW, cosine scheduler, warmup 5% |
| **Precision** | bfloat16 mixed precision |
| **Inference** | Logprob extraction (forward pass) |
| **TTA** | Answer-choice shuffling (4 permutations) |
| **Framework** | PyTorch, HuggingFace Transformers, PEFT |
| **GPU** | NVIDIA H100 80GB |

---

## 9. 시도하지 않은 / 향후 개선 가능 전략

- [ ] **더 큰 모델** (13B, 72B) — VRAM 한계로 미시도
- [ ] **다른 VLM** (InternVL, LLaVA) — 앙상블 다양성 확보
- [ ] **데이터 증강** — 이미지 augmentation (회전, 색조 변환 등)
- [ ] **Cross-validation 기반 앙상블** — k-fold로 다양성 확보
- [ ] **Prompt 변형 앙상블** — 다른 system prompt로 다양성 확보
- [ ] **OCR 보조** — 재활용품 라벨 텍스트 인식 추가 입력
- [ ] **Chain-of-Thought** — 답 도출 과정을 명시적으로 유도
