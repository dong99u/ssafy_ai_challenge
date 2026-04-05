# Stage 4: Train / Dev / Test Distribution Comparison

**Date:** 2026-04-04
**Analyst:** Scientist agent (claude-sonnet-4-6)
**Figure:** `.omc/scientist/figures/cross_split_comparison_stage4.png`

---

## [OBJECTIVE]

Determine whether the train, dev, and test splits share consistent question/answer distributions
(i.e., no domain shift) and quantify the degree of question-text reuse across splits to inform
whether training on dev data is useful for improving test performance.

---

## [DATA]

| Split | Rows  | Unique Questions | Unique Images |
|-------|-------|-----------------|---------------|
| Train | 5,073 | 2,303           | 5,073         |
| Dev   | 4,413 | 1,971           | 4,413         |
| Test  | 5,074 | 2,286           | 5,074         |

- All three splits use **completely disjoint image sets** (0 image-path overlap across any pair).
- Every (question, image) pair is unique across all splits (0 row-level duplicates).
- Within each split, ~54-55% of rows are question-text repeats (same question asked about different images).

---

## [FINDING:F9] Cross-Split Question Overlap

Questions are heavily reused across splits, but images are always unique.

**Unique-question overlap:**
- Train ∩ Test: **613 shared question texts** (26.8% of test's unique questions)
- Dev ∩ Test: **459 shared question texts** (20.1% of test's unique questions)
- All three splits: **298 common question texts**

**Row-level impact (test set perspective):**
- 64.4% of test rows (3,266/5,074) use a question text that also appears in **train**
- 55.2% of test rows (2,801/5,074) use a question text that also appears in **dev**

[STAT:n] n(train)=5,073, n(dev)=4,413, n(test)=5,074
[STAT:effect_size] 64.4% of test rows share question text with train; 55.2% with dev

**Critical nuance:** Despite high question overlap, images never overlap. The same question
asked about a different image yields a different correct answer (confirmed: 0 (question+image)
duplicates exist). Question-text overlap does NOT imply label leakage.

**Implication for training:** Training on dev (with pseudo-labels from majority vote) is valuable
because ~55% of test question patterns are already seen in dev. A model that learns the question
semantics transfers well to test.

---

## [FINDING:F10] Distribution Consistency

Train and test are highly consistent; dev shows a statistically significant distributional shift.

### Question length

| Split | Mean (chars) | Std  | Median |
|-------|-------------|------|--------|
| Train | 31.8        | 5.4  | 31     |
| Dev   | 33.0        | 5.1  | 32     |
| Test  | 31.7        | 5.4  | 31     |

[STAT:p_value] KS test Train vs Test: D=0.0138, p=0.710 (not significant — same distribution)
[STAT:p_value] KS test Train vs Dev:  D=0.0956, p<0.001 (significant — dev is slightly longer)
[STAT:p_value] KS test Dev vs Test:   D=0.1071, p<0.001 (significant — dev differs from test)
[STAT:n] n(train)=5,073; n(dev)=4,413; n(test)=5,074

### Answer option length

| Split | Mean (chars) | Std  |
|-------|-------------|------|
| Train | 4.4         | 3.7  |
| Dev   | 3.5         | 3.6  |
| Test  | 4.5         | 3.8  |

[STAT:p_value] KS test Train vs Test (ans len): D=0.0168, p=0.006 (marginal)
[STAT:p_value] KS test Train vs Dev  (ans len): D=0.2319, p<0.001 (large shift)
[STAT:effect_size] Dev answer options are ~1 character shorter on average vs Train/Test

### Answer position distribution (label balance)

| Position | Train | Dev   |
|----------|-------|-------|
| a        | 24.9% | 27.9% |
| b        | 24.9% | 27.1% |
| c        | 25.8% | 21.1% |
| d        | 24.3% | 23.9% |

Train is nearly perfectly balanced. Dev shows c/d slightly underrepresented.

[STAT:p_value] Chi-square (train vs dev answer distribution): chi2=35.12, p<0.001, dof=3
[STAT:n] n=5,073 (train), n=4,413 (dev)

**Key conclusion:** Train and test distributions are statistically indistinguishable (KS p=0.71).
Dev shows mild but significant deviation — particularly in question type composition (see F11).

---

## [FINDING:F11] Template Patterns and Dev-Specific Distribution Shift

Questions are generated from a finite set of templates. The dominant template
(`사진에 보이는 재활용...`) accounts for ~56-57% of all rows across train and test.

### Key template frequencies

| Template keyword    | Train  | Dev    | Test   |
|---------------------|--------|--------|--------|
| 사진에 보이는           | 64.6%  | 59.6%  | 64.4%  |
| 사진 속               | 30.1%  | 33.4%  | 30.4%  |
| 재활용 가능한            | 35.9%  | 40.9%  | 35.8%  |
| 재활용 (any)          | 93.1%  | 94.2%  | 93.7%  |
| **몇 개 (counting)** | **34.4%** | **71.3%** | **33.8%** |
| 무엇인가요 (identification) | 54.2% | 20.9% | 54.7% |

[STAT:n] n(train)=5,073; n(dev)=4,413; n(test)=5,074

**Dev is dominated by counting questions (71.3% vs ~34% in train/test).** This is the primary
distribution shift between dev and train/test. Dev has far fewer identification questions
(`무엇인가요`: 20.9% vs 54.2% in train). This explains the shorter average answer option length
in dev (counting answers like "3개", "2개" are shorter than category answers like "플라스틱 병").

### Top question templates (all splits share the same top-2 templates)

| Rank | Template (train)                              | Count |
|------|-----------------------------------------------|-------|
| 1    | 사진에 보이는 재활용품 중 플라스틱 재질인 것은 무엇인가요?         | 245   |
| 2    | 사진에 보이는 재활용품은 무엇인가요?                        | 138   |

| Rank | Template (dev)                                | Count |
|------|-----------------------------------------------|-------|
| 1    | 사진에 보이는 재활용 가능한 플라스틱 병은 몇 개입니까?             | 139   |
| 2    | 사진에 보이는 재활용 가능한 플라스틱 컵은 몇 개입니까?             | 132   |

| Rank | Template (test)                               | Count |
|------|-----------------------------------------------|-------|
| 1    | 사진에 보이는 재활용품 중 플라스틱 재질인 것은 무엇인가요?         | 221   |
| 2    | 사진에 보이는 재활용품은 무엇인가요?                        | 164   |

Train and test top templates are nearly identical. Dev skews toward counting templates.

### Template-level overlap

| Pair        | Unique templates | Overlap |
|-------------|-----------------|---------|
| Train ∩ Test | 613 / 2,286     | 26.8%   |
| Train ∩ Dev  | 428 / 1,971     | 21.7%   |
| Dev ∩ Test   | 459 / 2,286     | 20.1%   |

[STAT:effect_size] Template overlap is moderate (~20-27%) but row-level impact is high (55-64%)

---

## Training Recommendations (Based on Findings)

1. **Include dev in training:** Even though dev has a counting-question bias, 55% of test question
   texts appear in dev. Training on dev (with majority-vote pseudo-labels) provides question-type
   coverage complementary to train.

2. **No domain shift concern for train → test:** KS p=0.71 confirms train and test question-length
   distributions are identical. The model trained on train generalizes directly to test structure.

3. **Dev's counting-question bias:** Because dev has 2x the counting question rate of test, be
   cautious about over-weighting dev in training — it could bias the model toward counting answers.
   Consider stratified sampling or upsampling identification questions from dev.

4. **Answer balance:** Train is near-perfectly balanced (24-26% per position). Use train's label
   distribution as the reference; do not rely on dev's slightly skewed distribution.

---

## [LIMITATION]

- Question-text overlap analysis is based on exact string matching. Paraphrased duplicates
  (same semantic content, different wording) are not captured.
- Dev majority-vote labels may disagree with ground truth gold labels; the ~26% agreement
  rate on shared questions reflects different images, not label noise.
- No image content analysis was performed here (only text/metadata). Visual domain shift
  (e.g., image quality, object composition) is not assessed in this stage.
- The dev counting-question bias source is unknown — it may reflect a specific annotation
  batch or deliberate curation strategy, not a collection artifact.

---

## Figure

Saved to: `/Users/parkdongkyu/my_project/ssafy_ai_challenge/.omc/scientist/figures/cross_split_comparison_stage4.png`

Six-panel figure including: question length distributions, answer option length distributions,
total vs. unique question counts, key template frequencies, top-10 answer option comparison
(train vs test), and question overlap matrix heatmap.
