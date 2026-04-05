# EDA Research Report: SSAFY Recycling VQA Competition

**Session ID:** eda-vqa-20260404
**Date:** 2026-04-04
**Status:** Complete (5/5 stages)
**Goal:** Data insights to push accuracy from 0.87 → 0.93+

---

## Executive Summary

This EDA reveals 6 critical insights that directly inform the training strategy:

1. **Counting questions dominate** (51.6% overall, 34.5% in train) and "2개" is the single most common correct answer (13.5%). Counting accuracy is the #1 lever.
2. **Dev set has severe distribution bias** — 71.3% counting questions vs 34.5% in train. Blindly adding dev data will over-train on counting at the expense of identification questions.
3. **Dev annotations are extremely noisy** — Fleiss' κ=0.164 (slight agreement), 0% unanimous, only 20.7% with ≥4/5 agreement. Use ≥3/5 threshold (3,430 rows) with stratified sampling.
4. **IMAGE_SIZE=384 is too aggressive** — median native resolution is 720×960. Upgrading to 672px retains 3.1× more detail with manageable compute cost on H100.
5. **Train and test distributions are identical** (KS p=0.710) — no domain shift concern.
6. **Answer positions are perfectly balanced** in train (chi² p=0.494) — no position bias to exploit or worry about.

---

## Key Findings by Stage

### Stage 1: Answer Bias + Question Types

| Metric | Value |
|--------|-------|
| Answer position balance | Uniform (p=0.494) |
| Most common answer | "2개" (13.5%) |
| Counting questions (train) | 34.5% |
| Material questions (train) | 32.0% |
| Object ID questions (train) | 29.7% |
| Counting rank-2 bias | 50.8% correct = 2nd smallest number |

**Implication:** The model MUST be good at counting objects. A counting-aware system prompt and CoT reasoning will help.

### Stage 2: Annotator Agreement + Pseudo-Labels

| Metric | Value |
|--------|-------|
| Fleiss' κ | 0.164 (slight) |
| 5/5 unanimous | 0.0% |
| 4/5 agreement | 20.7% (891 rows) |
| 3/5 agreement | 59.0% (2,539 rows) |
| ≤2/5 tied | 20.3% (871 rows) |
| Usable pseudo-labels (≥3/5) | 3,430 rows |

**Implication:** Dev pseudo-labels are noisy. Use ≥3/5 threshold with stratified sampling matching train distribution. Consider model re-labeling for tied rows.

### Stage 3: Image Characteristics

| Metric | Value |
|--------|-------|
| Median resolution | 720×960 |
| Aspect ratio | 79% portrait, 16% landscape, 5% square |
| Cross-split consistency | >99% |
| Recommended IMAGE_SIZE | **672px** (3.1× detail vs 384px) |
| Alternative IMAGE_SIZE | 1024px (if H100 memory allows) |

**Implication:** 384→672 is the easiest accuracy win — just changing one number preserves 3× more visual information.

### Stage 4: Cross-Split Comparison

| Metric | Value |
|--------|-------|
| Train↔Test distribution | Identical (KS p=0.710) |
| Test question overlap with train | 64.4% row-level |
| Image overlap | 0% (completely disjoint) |
| Dev counting bias | 71.3% vs train 34.5% |

**Implication:** No domain shift between train and test. But dev has a massive counting bias — MUST stratify when using dev for training.

### Stage 5: Question NLP + Prompt Engineering

| Metric | Value |
|--------|-------|
| "재활용" in questions | 93.6% |
| Counting questions overall | 51.6% |
| Unique question patterns | 40.5% (highly templated) |
| Hardest category | Location questions (κ=0.580) |
| Hardest materials | Metal (0.592), Plastic (0.596) |

**Prompt Engineering Must-Haves:**
1. Korean recycling categories (플라스틱, 유리, 금속/캔, 종이, 비닐, 스티로폼)
2. Counting strategy: "Count only clearly visible objects. Enumerate before answering."
3. Metal vs. plastic disambiguation (cylindrical metallic vs transparent/colored plastic)
4. Location reference frame (viewer's perspective)
5. CoT trigger for counting questions

---

## Actionable Strategy

### Data Strategy
```
Training data composition:
├── train.csv: 5,073 rows (all, primary)
├── dev.csv (≥3/5 agreement): ~3,430 rows (stratified to match train distribution)
│   ├── Counting: subsample from 71.3% → ~34.5% weight
│   ├── Material ID: upsample from 13.1% → ~32% weight  
│   └── Object ID: upsample from 11.6% → ~30% weight
└── Total: ~7,500-8,500 effective training rows
```

### Model Strategy
```
Model: Qwen2.5-VL-7B-Instruct (proven at 0.87)
├── IMAGE_SIZE: 672 (was 384)
├── LoRA: r=32, alpha=64 (was r=16, alpha=32)
├── Epochs: 3 with cosine schedule (was 2 with linear)
├── LR: 2e-5 (keep) or try 1e-5
├── Inference: Logprob ranking (was text generation + parsing)
└── Prompt: Domain-specific Korean recycling expert
```

### Prompt Strategy
```python
SYSTEM_INSTRUCT = (
    "당신은 재활용 폐기물 분류 전문가입니다. "
    "이미지를 주의 깊게 관찰하고 질문에 정확히 답하세요. "
    "재활용품 종류: 플라스틱, 유리, 금속/캔, 종이/골판지, 비닐, 스티로폼. "
    "개수를 세는 질문은 눈에 보이는 물체만 하나씩 세세요. "
    "반드시 a, b, c, d 중 하나의 소문자 한 글자로만 답하세요."
)
```

### Estimated Impact
| Change | From | To | Expected Gain |
|--------|------|-----|---------------|
| Add dev data (stratified) | 5,073 rows | ~8,000 rows | +0.03~0.04 |
| IMAGE_SIZE 384→672 | 20% detail | 63% detail | +0.02~0.04 |
| Logprob inference | Text parsing | Direct logprob | +0.01 |
| Domain prompt | Generic EN | Korean recycling | +0.01~0.02 |
| LoRA r=32 + 3 epochs | r=16, 2 ep | r=32, 3 ep | +0.01~0.02 |
| **Total** | **0.87** | | **+0.08~0.13 → 0.95~1.00** |
| **Realistic estimate** | **0.87** | | **0.93~0.96** |

---

## Limitations

1. Dev labels have no gold standard — even majority vote carries ~35% noise
2. Image-level difficulty analysis not performed (would require running model and comparing)
3. Test set answer distribution is unknown (only question distribution compared)
4. Counting rank-2 bias may be dataset-specific, not generalizable to test
5. Prompt optimization is theoretical — needs empirical validation

---

## Figures

All figures saved to `.omc/scientist/figures/`:
- `stage1_eda_overview.png` — Answer distribution + question types
- `stage1_eda_details.png` — Counting bias analysis
- `01_agreement_levels.png` — IAA distribution
- `02_annotator_comparison.png` — Pairwise annotator agreement
- `03_difficulty_distribution.png` — Agreement by question type
- `04_pseudo_label_strategy.png` — Strategy comparison
- `image_resolution_distribution.png` — Resolution statistics
- `aspect_ratio_analysis.png` — Aspect ratio distribution
- `cross_split_comparison_stage4.png` — Split comparison
- `keyword_frequency.png` — Question keyword analysis
- `question_distribution_eda.png` — Question type distribution
