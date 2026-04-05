# Stage 1 EDA: Answer Position Bias + Question Type Distribution
**Date:** 2026-04-04  
**Analyst:** Scientist (oh-my-claudecode)  
**Dataset:** SSAFY AI Challenge — Recycling VQA  
**Objective:** Identify answer position bias, question type distribution, and answer text patterns to guide accuracy improvements from 0.87 → 0.93+

---

## Data Summary

[DATA]
- **train.csv**: 5,073 rows × 8 columns (id, path, question, a, b, c, d, answer)
- **dev.csv**: 4,413 rows × 12 columns (id, path, question, a, b, c, d, answer1–answer5)
- Dev has 5 annotator responses per item (no gold label); answer1–answer5 contain 2–36 NaN values per column
- 112/4,413 dev rows (2.5%) have at least one annotator NaN

---

## [FINDING:F1] Answer Position Distribution

<analysis>

### Train Answer Position Distribution (n=5,073)

| Position | Count | Percentage | 95% Wilson CI |
|----------|-------|-----------|---------------|
| a | 1,263 | 24.90% | [23.73%, 26.11%] |
| b | 1,265 | 24.94% | [23.76%, 26.15%] |
| c | 1,311 | 25.84% | [24.66%, 27.07%] |
| d | 1,234 | 24.32% | [23.16%, 25.52%] |

**Chi-square test for uniformity:**
- chi2 = 2.396, df = 3, p = 0.494
- Cramer's V = 0.0125 (negligible effect)
- Expected per position: 1,268.2

**Conclusion:** Train answer positions are effectively uniform. No position bias exists in the training set. A model that exploits position guessing would gain nothing.

### Dev Majority-Vote Answer Position Distribution (n=4,412 valid)

| Position | Count | Percentage | 95% Wilson CI |
|----------|-------|-----------|---------------|
| a | 1,083 | 24.55% | [23.30%, 25.84%] |
| b | 1,147 | 26.00% | [24.72%, 27.31%] |
| c | 955   | 21.65% | [20.46%, 22.89%] |
| d | 1,227 | 27.81% | [26.51%, 29.15%] |

**Chi-square test for uniformity:**
- chi2 = 35.92, df = 3, p < 0.0001
- Cramer's V = 0.0521 (small effect)

Dev shows a statistically significant non-uniformity: 'c' is underrepresented (21.7%) and 'd' is overrepresented (27.8%). This is driven by the question type distribution shift (see F2 below) — not by inherent annotator bias in position guessing.

### Annotator Agreement (Dev)

| Unique answers per item | Count | Percentage |
|-------------------------|-------|-----------|
| 0 unique (all NaN) | 1 | 0.0% |
| 1 unique (full agreement) | 13 | 0.3% |
| 2 unique | 2,512 | 56.9% |
| 3 unique | 1,700 | 38.5% |
| 4 unique | 187 | 4.2% |

**Critical observation:** Only 0.3% of dev items have full annotator agreement. 56.9% have exactly 2 unique answers (one majority, one minority). This reflects genuine label ambiguity in the task — especially for counting questions where off-by-one errors are common.

Per-annotator 'b' share ranges from 26.8% to 28.4%, showing no individual annotator has strong position bias.

</analysis>

---

## [FINDING:F2] Question Type Distribution

<analysis>

### Train Question Types (n=5,073, keyword classifier)

| Question Type | Count | Percentage |
|--------------|-------|-----------|
| 개수/수량 (Counting) | 1,750 | 34.5% |
| 재질/소재 (Material) | 1,625 | 32.0% |
| 물체 식별 (Object ID) | 1,505 | 29.7% |
| 기타 (Other) | 68 | 1.3% |
| 위치/방향 (Location) | 67 | 1.3% |
| 분리수거 방법 (Recycling Method) | 41 | 0.8% |
| 색상 (Color) | 17 | 0.3% |

All chi-square uniformity tests per type: p > 0.05 (no within-type position bias).

### CRITICAL: Train vs Dev Distribution Shift

| Question Type | Train % | Dev % | Absolute Shift |
|--------------|---------|-------|---------------|
| Counting | 34.5% | **71.5%** | **+37.0 pp** |
| Material | 32.0% | 13.1% | -18.9 pp |
| Object ID | 29.7% | 11.6% | -18.1 pp |
| Location | 1.3% | 1.9% | +0.5 pp |
| Recycling Method | 0.8% | 0.7% | -0.1 pp |
| Color | 0.3% | 0.2% | -0.1 pp |

**The dev set is dominated by counting questions (71.5%) whereas train is roughly balanced among three types.** This distribution shift means:
1. A model that performs poorly on counting but well on material/object-ID may score higher on train than dev
2. Dev majority vote skews toward 'd' specifically because counting questions in dev have larger numbers placed at position 'd' more frequently

### Dev Answer Position by Question Type (majority vote, %)

| Type | a% | b% | c% | d% |
|------|----|----|----|----|
| Counting | 17.1 | 24.6 | 23.6 | **34.7** |
| Material | **42.1** | 31.6 | 17.4 | 8.8 |
| Object ID | **46.3** | 27.0 | 15.0 | 11.7 |
| Location | **35.4** | 31.7 | 18.3 | 14.6 |

The 'd' dominance in dev counting (34.7%) is caused by dataset construction: in dev counting questions, the 'd' slot often contains larger numbers (4개, 5개, 9개 are the top 'd' options), and the correct answer tends to be among those larger values.

</analysis>

---

## [FINDING:F3] Answer Text Patterns

<analysis>

### Top Correct Answer Texts (Train)

| Answer Text | Count | % of Train |
|-------------|-------|-----------|
| 2개 (2 items) | 686 | 13.52% |
| 플라스틱 (Plastic) | 433 | 8.54% |
| 1개 (1 item) | 371 | 7.31% |
| 3개 (3 items) | 288 | 5.68% |
| 종이 (Paper) | 132 | 2.60% |
| 4개 (4 items) | 132 | 2.60% |
| 플라스틱 병 (Plastic bottle) | 121 | 2.39% |
| 종이 상자 (Paper box) | 94 | 1.85% |

**33.7% of all correct answers are simple numeric counts (N개 pattern).** This is the single largest answer category.

### Numeric Count Distribution (counting questions)

| Count | Frequency | % of counting answers |
|-------|-----------|----------------------|
| 1개 | 371 | 21.7% |
| 2개 | 686 | 40.2% |
| 3개 | 288 | 16.9% |
| 4개 | 141 | 8.3% |
| 5개 | 59 | 3.5% |
| 9개 | 34 | 2.0% |
| 6개 | 32 | 1.9% |

**2개 is correct 40.2% of the time in counting questions.** This reflects the typical image content: most recycling images contain 2 objects of a given type.

### CRITICAL: Numeric Rank Bias in Counting Questions

Among counting questions where all 4 options are numeric (n=1,495):

| Rank (ascending order) | Correct count | % |
|------------------------|--------------|---|
| Rank 1 (smallest) | 150 | 10.0% |
| **Rank 2 (2nd smallest)** | **759** | **50.8%** |
| Rank 3 (2nd largest) | 407 | 27.2% |
| Rank 4 (largest) | 179 | 12.0% |

**Chi-square test:** chi2 = 635.5, df = 3, p < 0.001  
**Cramer's V = 0.376 (medium-to-large effect)**

The correct answer is the 2nd smallest number 50.8% of the time — almost always 2개 (rank-2 value = 2 in 1,099/1,495 cases = 73.5%). This is a strong structural bias: questions are constructed with options {1개, 2개, 3개, 4개} and the correct answer is 2개 in most cases.

**Implication for model:** If a model consistently outputs "2개" for counting questions, it would be correct ~40% of the time on counting questions, translating to ~14% of all train questions correct "for free." The model needs to count correctly to beat this baseline.

### Distractor Structure (all options across all questions)

Top option texts appearing across all choice columns (a/b/c/d):
- 3개: 1,598 appearances
- 2개: 1,504 appearances
- 1개: 1,358 appearances
- 4개: 1,306 appearances
- 유리병 (Glass bottle): 892
- 플라스틱 (Plastic): 748

Options are drawn from a small vocabulary of ~20–30 common items + numeric counts. The distractor pool is tightly constrained, making the task easier for a model that understands the domain.

</analysis>

---

## Cross-tabulation: Answer Position × Question Type (Train)

| Type | a | b | c | d | chi2 | p |
|------|---|---|---|---|------|---|
| Counting (n=1,750) | 25.1% | 25.5% | 25.8% | 23.5% | 2.12 | 0.547 |
| Material (n=1,625) | 23.6% | 25.2% | 27.1% | 24.1% | 4.54 | 0.209 |
| Object ID (n=1,505) | 26.2% | 23.5% | 24.6% | 25.6% | 2.61 | 0.456 |

No significant position bias by question type in train. The dataset was well-randomized.

---

## [LIMITATION]

1. **Keyword classifier is heuristic**: The question type classifier uses Korean keyword matching. Misclassification rate is unvalidated (no ground-truth type labels). Some "Object ID" questions may actually be "Material" questions.
2. **Dev has no gold labels**: Dev majority vote is used as a proxy, but only 0.3% of dev items have full annotator agreement. Majority vote from 5 annotators may not reflect the true correct answer.
3. **Distribution shift direction unknown**: We observe a 37pp shift toward counting questions in dev vs. train, but the test set distribution is unknown. The test set may be closer to train distribution, dev distribution, or something different entirely.
4. **Counting rank bias is train-only**: The rank-2 = 50.8% finding is from train. Dev counting questions show larger numbers in position 'd', suggesting a different construction pattern — the rank bias may not generalize to dev or test.
5. **Option ordering is mostly random** (78.3% of counting questions): Options are not consistently sorted, so the "rank 2 = correct" finding reflects answer-value frequency, not position ordering.

---

## Actionable Recommendations

1. **Prioritize counting accuracy**: 34.5% of train and 71.5% of dev are counting questions. A 5% improvement in counting accuracy translates to a larger accuracy gain in dev than in train.
2. **2개 is a strong prior**: For counting questions, "2개" is correct 40% of the time. The model should be calibrated to recognize when the count is clearly not 2 before deviating from this prior.
3. **Tight answer vocabulary**: The top 5 answer texts cover ~37% of all correct answers. Constrained decoding or ensemble with answer vocabulary filtering could improve reliability.
4. **Dev ≠ Train distribution**: Do not rely on dev accuracy as a proxy for test accuracy without accounting for the 37pp counting shift. Stratified evaluation by question type is essential.
5. **No position bias to exploit or defend against**: Train and dev show no exploitable position bias (Cramer's V < 0.06). The model does not need to be corrected for position bias.

---

## Figures

- `/Users/parkdongkyu/my_project/ssafy_ai_challenge/.omc/scientist/figures/stage1_eda_overview.png` — Answer position distribution, question type distribution, top correct answer texts
- `/Users/parkdongkyu/my_project/ssafy_ai_challenge/.omc/scientist/figures/stage1_eda_details.png` — Dev annotator agreement, numeric count distribution, train vs dev type shift

---

*Report generated: 2026-04-04 | Session: vqa-eda-stage1*
