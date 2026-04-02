# VQA Challenge Dataset - Comprehensive Statistical Analysis

**Analysis Date:** 2026-04-02  
**Dataset:** SSAFY VQA Challenge (Korean Visual Question Answering)  
**Language:** Korean (한국어)  
**Task Type:** Multiple Choice Question Answering (4 options: a, b, c, d)

---

## 1. DATASET OVERVIEW

### Data Splits Summary

| Split | Rows | Columns | Images | Label Type |
|-------|------|---------|--------|-----------|
| **Train** | 5,073 | 8 | 5,073 | Single (answer) |
| **Test** | 5,074 | 7 | 5,074 | None (evaluation) |
| **Dev** | 4,413 | 12 | 4,413 | Multiple (5 annotators) |
| **Total** | **14,560** | - | **14,560** | - |

### Data Quality Assessment

[FINDING] All 14,560 images are present and match CSV row counts perfectly
[STAT:n] Train: 5,073/5,073 (100%), Test: 5,074/5,074 (100%), Dev: 4,413/4,413 (100%)

[FINDING] Train and Test splits have zero missing values; Dev has minimal missing data
[STAT:missing] Train: 0 cells (0.00%), Test: 0 cells (0.00%), Dev: 118 cells (0.22%)
[STAT:detail] Dev missing values concentrated in multiple answer columns (answer1-5), suggesting some ambiguous/difficult samples

[FINDING] All sample IDs are unique across all splits; no duplicate ID conflicts
[STAT:duplicates] Train: 0 duplicates, Test: 0 duplicates, Dev: 0 duplicates

---

## 2. QUESTION ANALYSIS

### Question Reuse Across Dataset

[FINDING] ~45% of questions appear multiple times across train/test/dev (question reuse pattern)
[STAT:n] Train: 2,303 unique of 5,073 total (45.4% reuse), Test: 2,286 unique of 5,074 total (45.0% reuse), Dev: 1,971 unique of 4,413 total (44.6% reuse)

This reuse pattern is expected in VQA datasets where similar questions are asked about different images or vice versa.

### Question Text Length Distribution

#### Train Questions
- **Length Range:** 17–73 characters
- **Mean:** 31.8 chars
- **Median:** 31.0 chars
- **Std Dev:** 5.4 chars
[STAT:n] n = 5,073

#### Test Questions
- **Length Range:** 17–67 characters
- **Mean:** 31.7 chars
- **Median:** 31.0 chars
- **Std Dev:** 5.4 chars
[STAT:n] n = 5,074

#### Dev Questions
- **Length Range:** 17–71 characters
- **Mean:** 33.0 chars
- **Median:** 32.0 chars
- **Std Dev:** 5.1 chars
[STAT:n] n = 4,413

[FINDING] Question lengths are highly consistent across all splits (mean ~31–33 chars)
[STAT:ci] 95% of questions fall within 20–46 character range (approx ±2 SD from mean)

---

## 3. ANSWER OPTION ANALYSIS

### Answer Option Text Length Distribution

#### Train Set
| Option | Min | Max | Mean | Median |
|--------|-----|-----|------|--------|
| a | 1 | 42 | 4.4 | 3 |
| b | 1 | 34 | 4.4 | 3 |
| c | 1 | 33 | 4.4 | 3 |
| d | 1 | 40 | 4.5 | 3 |

#### Test Set
| Option | Min | Max | Mean | Median |
|--------|-----|-----|------|--------|
| a | 1 | 30 | 4.5 | 3 |
| b | 1 | 40 | 4.6 | 3 |
| c | 1 | 33 | 4.5 | 3 |
| d | 1 | 33 | 4.5 | 3 |

#### Dev Set
| Option | Min | Max | Mean | Median |
|--------|-----|-----|------|--------|
| a | 1 | 43 | 3.7 | 2 |
| b | 1 | 43 | 3.6 | 2 |
| c | 1 | 43 | 3.5 | 2 |
| d | 1 | 43 | 3.4 | 2 |

[FINDING] Answer option lengths are short and balanced across options (mean 3.5–4.6 chars)
[STAT:observation] Train/Test options slightly longer (mean 4.4–4.6) than Dev (mean 3.4–3.7), suggesting different answer annotation strategies

---

## 4. ANSWER DISTRIBUTION & CLASS BALANCE

### Train Set Label Distribution

| Answer | Count | Percentage |
|--------|-------|-----------|
| a | 1,263 | 24.9% |
| b | 1,265 | 24.9% |
| c | 1,311 | 25.8% |
| d | 1,234 | 24.3% |

[FINDING] Train set answers are well-balanced across all four options
[STAT:chi_square] Chi-square test: All categories within 2.5% of expected 25% (uniform distribution)
[STAT:effect] No significant class imbalance detected (balanced dataset)

### Dev Set Multiple Annotations

**answer1 Distribution:**
- a: 1,054 (23.9%), b: 1,254 (28.4%), c: 1,027 (23.3%), d: 1,076 (24.4%)

**answer2 Distribution:**
- a: 1,056 (23.9%), b: 1,185 (26.9%), c: 1,050 (23.8%), d: 1,103 (25.0%)

**answer3 Distribution:**
- a: 1,091 (24.7%), b: 1,190 (27.0%), c: 1,043 (23.6%), d: 1,055 (23.9%)

**answer4 Distribution:**
- a: 1,037 (23.5%), b: 1,174 (26.6%), c: 1,065 (24.1%), d: 1,101 (24.9%)

**answer5 Distribution:**
- a: 1,014 (22.9%), b: 1,186 (26.9%), c: 1,045 (23.7%), d: 1,141 (25.8%)

[FINDING] Dev set shows slight systematic bias: answer 'b' is chosen ~26.8% (above 25%), others near 24%
[STAT:observation] Consistent across all 5 annotators, suggesting potential label bias or question formulation
[LIMITATION] Multi-annotator agreement rates not analyzed; inter-annotator agreement could affect model performance

---

## 5. MISSING DATA ANALYSIS

### Dev Set Missing Values

[FINDING] Dev set has 118 missing values (0.22%) concentrated in multiple answer columns
[STAT:missing_per_column]
- answer1: 2 missing (0.05%)
- answer2: 19 missing (0.43%)
- answer3: 34 missing (0.77%)
- answer4: 36 missing (0.82%)
- answer5: 27 missing (0.61%)

[LIMITATION] Missing annotations in answer2-5 suggest some samples were difficult to annotate or had consensus issues. Recommend imputation or exclusion of rows with >2 missing annotations.

---

## 6. DATA INTEGRITY CHECKS

### Image-CSV Alignment
- Train: 5,073 images ↔ 5,073 CSV rows ✓
- Test: 5,074 images ↔ 5,074 CSV rows ✓
- Dev: 4,413 images ↔ 4,413 CSV rows ✓

[FINDING] Perfect alignment between image files and CSV records across all splits
[STAT:n] 14,560/14,560 images verified (100%)

### Column Schema Consistency

**Train:** id, path, question, a, b, c, d, answer  
**Test:** id, path, question, a, b, c, d  
**Dev:** id, path, question, a, b, c, d, answer1, answer2, answer3, answer4, answer5

[FINDING] Column schemas are consistent and appropriate for the task structure

---

## 7. DATASET CHARACTERISTICS SUMMARY

| Metric | Value |
|--------|-------|
| **Total Samples** | 14,560 |
| **Total Images** | 14,560 |
| **Unique Questions** | ~6,560 |
| **Language** | Korean (한국어) |
| **Task Type** | Multiple Choice (4 options) |
| **Train/Test/Dev Split Ratio** | 1.15 : 1.15 : 1.0 |
| **Question Reuse Rate** | ~45% |
| **Class Balance (Train)** | Excellent (24–26%) |
| **Missing Data Rate** | 0.22% (Dev only) |
| **Data Quality** | Clean & Complete |

---

## 8. KEY FINDINGS & RECOMMENDATIONS

### Strengths
1. **Perfect data alignment:** All 14,560 images present and matched
2. **No missing values in main splits:** Train (0%), Test (0%) suitable for immediate use
3. **Balanced class distribution:** No class imbalance issues in training set
4. **Consistent question format:** ~32 char questions (standardized difficulty)
5. **Multiple annotations:** 5 diverse answers per sample in Dev for robust evaluation

### Limitations & Considerations

[LIMITATION] Dev set has ~0.2% missing annotations; recommend filtering rows with >1 missing answer before aggregation

[LIMITATION] Answer 'b' shows ~1.8 percentage point bias across Dev annotations; may indicate labeling preference or question construction bias

[LIMITATION] 45% question reuse suggests potential data leakage risk if similar questions appear across train/test; recommend explicit overlap verification

[LIMITATION] Inter-annotator agreement (IAA) statistics not computed; unknown how to weight/aggregate the 5 Dev answers

### Recommendations for Modeling

1. **Data preparation:**
   - Remove or impute 118 missing answers in Dev set (consider majority voting from 4 other annotations)
   - Verify no explicit train/test/dev question overlap at substring level

2. **Baseline metrics:**
   - Majority class baseline: ~25% (chance performance) for 4-way classification
   - Target: Early stopping benchmark should exceed 70% train/dev accuracy

3. **Validation strategy:**
   - Use stratified k-fold on Train (preserve 24–26% class balance)
   - Use Dev for final evaluation with majority voting across 5 annotations

4. **Model considerations:**
   - Short answers (3–4 chars) → vocabulary limited; embedding-based models recommended
   - Multi-lingual VQA (Korean) → use multilingual vision encoders (CLIP, BLIP-2 multilingual variants)

---

## 9. FILES & PATHS

- **Statistics JSON:** `/Users/parkdongkyu/my_project/ssafy_ai_challenge/.omc/scientist/reports/vqa_dataset_statistics.json`
- **Train CSV:** `/Users/parkdongkyu/my_project/ssafy_ai_challenge/data/csv/train.csv`
- **Test CSV:** `/Users/parkdongkyu/my_project/ssafy_ai_challenge/data/csv/test.csv`
- **Dev CSV:** `/Users/parkdongkyu/my_project/ssafy_ai_challenge/data/csv/dev.csv`
- **Image Directories:** `/Users/parkdongkyu/my_project/ssafy_ai_challenge/data/{train,test,dev}/`

---

**Analysis Completed:** 2026-04-02 | **Analysis by:** Scientist Agent
