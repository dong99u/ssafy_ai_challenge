# VQA Answer Distribution & Bias Analysis
**Report generated:** 2026-04-03
**Dataset:** SSAFY AI Challenge — VQA Competition
**Analyst:** Scientist agent (oh-my-claudecode)

---

## [OBJECTIVE]
Identify answer position biases and text-level patterns in train.csv and dev.csv
that could be exploited to improve competition accuracy beyond pure vision-language modelling.

---

## [DATA]
| Split | Rows | Columns | Answer columns |
|-------|------|---------|----------------|
| train | 5,073 | 8 | `answer` (single label: a/b/c/d) |
| dev   | 4,413 | 12 | `answer1`–`answer5` (5 annotators) |

Question types detected (train):
- Counting (몇): 1,755 (34.6%)
- Material/Type (재질/분류/종류): 2,224 (43.8%)
- Color (색): 473 (9.3%)
- What/Other (무엇 등): 621 (12.2%)

---

## Analysis Results

### 1. Overall Answer Position Distribution — Train

| Position | Count | % |
|----------|-------|---|
| a | 1,263 | 24.90% |
| b | 1,265 | 24.94% |
| c | 1,311 | 25.84% |
| d | 1,234 | 24.32% |

[FINDING] Train answer positions are uniformly distributed — no exploitable position bias.
[STAT:p_value] chi2 = 2.40, p = 0.494 (df=3) — NOT significant
[STAT:n] n = 5,073

---

### 2. Dev Majority-Vote Distribution

| Position | Count | % |
|----------|-------|---|
| a | 1,083 | 24.54% |
| b | 1,147 | 25.99% |
| c |   955 | 21.64% |
| d | 1,227 | 27.80% |

[FINDING] Dev majority-vote distribution is SIGNIFICANTLY non-uniform, with position D over-represented (+3.49%) and C under-represented (−4.20%) relative to train.
[STAT:p_value] chi2 = 35.92, p < 0.001 (df=3) — SIGNIFICANT
[STAT:n] n = 4,413

[FINDING] Train and Dev distributions are statistically different — the dev evaluation set has a systematic D-position lean.
[STAT:p_value] chi2 (train vs dev-majority) = 65.90, p < 0.001

---

### 3. Per-Annotator Bias in Dev

All 5 annotators independently show a B-position preference:

| Annotator | a% | b% | c% | d% | chi2 | p |
|-----------|----|----|----|----|------|---|
| answer1 | 23.9 | 28.4 | 23.3 | 24.4 | 28.75 | <0.001 |
| answer2 | 24.0 | 27.0 | 23.9 | 25.1 | 10.62 | 0.014 |
| answer3 | 24.9 | 27.2 | 23.8 | 24.1 | 12.19 | 0.007 |
| answer4 | 23.7 | 26.8 | 24.3 | 25.2 |  9.63 | 0.022 |
| answer5 | 23.1 | 27.0 | 23.8 | 26.0 | 17.74 | <0.001 |

[FINDING] Every annotator shows statistically significant B-position bias (p < 0.05 for all 5 annotators). Position B is chosen ~27% vs 25% expected — a consistent human tendency to pick the second option.
[STAT:effect_size] B excess = ~+2 percentage points per annotator
[STAT:p_value] All annotators: p < 0.05
[STAT:n] n = 4,413 per annotator

---

### 4. Annotator Agreement in Dev

| Agreement | Count | % |
|-----------|-------|---|
| 5/5 (unanimous) | 1 | 0.0% |
| 4/5 | 902 | 20.4% |
| 3/5 (bare majority) | 2,606 | 59.1% |
| 2/5 | 903 | 20.5% |
| 1/5 | 1 | 0.0% |

[FINDING] Agreement is extremely low — 79.6% of dev questions have only bare-majority (3/5) or minority votes. This indicates high visual ambiguity in the dataset.
[STAT:effect_size] Mean agreement = 3.000/5 (barely above random majority)
[STAT:n] n = 4,413

---

### 5. Answer Text Analysis

Top correct answer texts (train):

| Text | Count | % of train |
|------|-------|-----------|
| 2개  | 686 | 13.52% |
| 플라스틱 | 433 | 8.54% |
| 1개  | 371 | 7.31% |
| 3개  | 288 | 5.68% |
| 종이  | 132 | 2.60% |
| 4개  | 132 | 2.60% |

[FINDING] '2개' is the single most common correct answer, accounting for 13.5% of all training labels. When '2개' appears as a choice, it is correct 45.6% of the time — far above the 25% baseline.
[STAT:effect_size] P(correct | '2개' is an option) = 45.6% vs 25.0% baseline
[STAT:n] n = 1,504 rows where '2개' appears as an option

Answer text distribution across positions is uniform — no text is systematically placed in a specific position (all top texts appear ~25% per position).

---

### 6. Critical Finding — Numeric Answer Value Rank Bias (Counting Questions)

For counting questions where all 4 options are numbers:

| Rank (sorted ascending) | Count | % |
|------------------------|-------|---|
| 1st (smallest) | 145 | 10.7% |
| **2nd** | **711** | **52.7%** |
| 3rd | 372 | 27.6% |
| 4th (largest) | 122 | 9.0% |

[FINDING] CRITICAL BIAS: When counting questions offer 4 numeric options, the correct answer is the 2nd smallest value in 52.7% of cases — more than double the uniform expectation of 25%. This is the strongest exploitable signal found in the dataset.
[STAT:p_value] chi2 = 612.6, p < 0.001 (df=3)
[STAT:effect_size] 52.7% vs 25.0% expected — effect = +27.7 percentage points
[STAT:n] n = 1,350 all-numeric counting rows

Mechanistic explanation: The dataset predominantly photographs scenes with 2 recyclable items; option sets are constructed as {1,2,3,4} or similar consecutive ranges, making 2 the 2nd-rank value in most cases.

---

### 7. Keyword-Group Position Bias

No significant position bias exists within any question keyword group in train:

| Group | n | chi2 | p | Dominant pos |
|-------|---|------|---|-------------|
| 몇 (counting) | 1,755 | 2.00 | 0.572 | c (25.8%) |
| 재질/분류/종류 | 2,224 | 1.67 | 0.645 | c (26.0%) |
| 색 (color) | 473 | 3.41 | 0.333 | c (28.3%) |
| 무엇 (what) | 2,809 | 0.80 | 0.851 | c (25.5%) |

[FINDING] No keyword group shows statistically significant position bias in train. The dataset curator successfully balanced position labels within each question type.
[STAT:p_value] All p > 0.05

---

## Summary of Exploitable Patterns

1. **Numeric rank bias (most actionable)**: For counting questions with all-numeric options, predict the 2nd smallest number as the default. Expected accuracy gain: 52.7% vs 25% baseline on this subset (~26.6% of train data).

2. **'2개' frequency**: If the model is uncertain on a counting question, '2개' as an outright prior gives 13.5% coverage at elevated P(correct) = 45.6%.

3. **Dev D-position lean**: Dev majority answers lean toward D (+3.49% relative to train). Marginal at best; train distribution should be the reference for model training.

4. **B-position annotator bias**: Individual annotators prefer B — but majority vote largely cancels this out. Not useful for final prediction.

---

## [LIMITATION]

1. Train answer balance is deliberate (all chi2 p > 0.49) — the dataset was curated for balance. Exploiting biases risks overfitting to dataset artifacts rather than visual understanding.
2. The numeric rank bias (2nd smallest) is extremely strong (chi2=612, p<0.001) but applies only to the ~26% of questions that are purely numeric counting questions.
3. Dev annotator agreement is very low (mean=3.0/5), meaning ~20% of dev labels are contested — evaluation noise is high.
4. Answer text position distribution is near-uniform; no position-specific text traps were found.
5. Analysis is purely statistical; causal mechanisms (e.g., why '2개' is so frequent) require domain knowledge about the recycling image collection process.

---

## Figures
- `fig1_answer_distribution.png` — Train vs Dev majority position distributions
- `fig2b_counting_rank_bias_extended.png` — Numeric rank bias in counting questions
- `fig3_annotator_position_bias.png` — Per-annotator B-position preference
- `fig4_correct_answer_texts.png` — Top correct answer texts
