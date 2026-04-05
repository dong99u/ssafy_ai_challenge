# Stage 2: Dev Set Annotator Agreement (IAA) + Pseudo-Label Quality Analysis

**Date**: 2026-04-04  
**Analyst**: Scientist Agent  
**Data**: `/data/csv/dev.csv` — 4,413 rows, 12 columns (id, path, question, a/b/c/d, answer1–5)

---

[OBJECTIVE] Determine inter-annotator agreement on dev.csv (5 annotators per question) and establish the optimal pseudo-label strategy for augmenting the training set.

[DATA]
- Total rows: 4,413
- Rows with all 5 annotators present: 4,301 (112 rows have ≥1 missing annotation)
- Missing annotations: answer1=2, answer2=19, answer3=34, answer4=36, answer5=27
- Answer options: always 4 choices (a/b/c/d), no null options
- Question types found: counting (71.7%), type identification (24.5%), spatial (3.3%), other (0.4%)
- All analysis below conducted on the 4,301 fully-annotated rows unless noted

---

## [FINDING:F4] Inter-Annotator Agreement Distribution

**Hypothesis**: Annotators show substantial agreement (kappa > 0.6) on visually clear recycling images.

**Result**: Hypothesis REJECTED. Agreement is only slight.

Key finding: **Not a single question received unanimous (5/5) agreement** across the entire dev set. The distribution is:

| Agreement Level | Count | % of dev |
|----------------|-------|----------|
| 4/5 (strong)   | 891   | 20.7%    |
| 3/5 (weak)     | 2,539 | 59.0%    |
| 2/5 (tied)     | 871   | 20.3%    |
| ≤1/5           | 0–1   | ~0%      |

[STAT:effect_size] Fleiss' Kappa = 0.1635 (slight agreement, per Landis & Koch 1977 scale)
[STAT:ci] Observed agreement P_bar = 37.3%; Expected by chance P_e = 25.1%
[STAT:n] n = 4,301 rows × 5 annotators = 21,505 annotations

**Pairwise annotator agreement** is nearly uniform across all 10 annotator pairs:
- Range: 0.351 to 0.393  
- Mean pairwise agreement: 37.3% ± 1.1%

[STAT:effect_size] Mean pairwise agreement = 0.373 (far below chance-adjusted 1.0, only marginally above chance baseline of 0.25)

The uniformity of pairwise agreement (SD=0.011) indicates no single "outlier annotator" — all annotators genuinely disagree on the inherent ambiguity of these recycling images.

**Figures**: `.omc/scientist/figures/01_agreement_levels.png`, `02_annotator_comparison.png`

[LIMITATION] Fleiss' kappa assumes equal weight for all annotation errors. Since this is a 4-class problem with uniform choice distribution (a=23.7%, b=27.3%, c=23.9%, d=25.0%), chance agreement ≈ 25%, making the low kappa especially striking. The 112 rows with missing annotations were excluded from kappa calculation, which may slightly underestimate true kappa.

---

## [FINDING:F5] Majority Vote Quality

**Hypothesis**: At least 50% of dev questions have a clear majority (>=4/5) suitable for high-confidence pseudo-labeling.

**Result**: Hypothesis REJECTED. Only 20.7% have clear majority.

Among the 4,301 fully-annotated rows:

| Category | n | % | Label noise estimate |
|---------|---|---|---------------------|
| Strong majority (4/5) | 891 | 20.7% | ~20% |
| Weak majority (3/5) | 2,539 | 59.0% | ~40% |
| Tied (2/2/1 or 2/1/1/1) | 871 | 20.3% | >50% |

[STAT:n] n = 4,301 (all fully-annotated dev rows)
[STAT:effect_size] Mean agreement ratio = 0.601 (i.e., average 3.0 of 5 annotators agree)

**Tie breakdown** (among the 871 tied rows):
- Pattern (2,2,1): 685 rows (78.6%) — two answers split the vote equally with one stray  
- Pattern (2,1,1,1): 186 rows (21.4%) — maximum fragmentation, minority plurality only

[STAT:p_value] Not applicable — this is a descriptive count analysis

**Answer position bias**: In high-agreement rows, answer 'd' dominates (38.2% of votes), while in low-agreement rows the distribution is nearly uniform (a=20.9%, b=27.6%, c=26.7%, d=24.8%). This suggests the test-makers may have intentionally placed the harder-to-count or ambiguous correct answer at position 'd' in clear cases.

**Question difficulty by type**:

| Type | n | Mean agreement | % High (>=4/5) | % Low (<=2/5) |
|------|---|---------------|---------------|---------------|
| Counting (몇 개) | 3,083 | 3.002 | 22.9% | 22.6% |
| Type ID (무엇) | 1,054 | 3.027 | 15.1% | 12.4% |
| Spatial (어디) | 144 | 2.903 | 15.3% | 25.0% |

- **Spatial questions are the hardest**: 25.0% low-agreement, only 15.3% high-agreement
- **Type identification questions are most reliable**: lowest low-agreement (12.4%)
- **Counting questions** dominate the dataset (71.7%) and show moderate difficulty

[STAT:n] Counting: n=3,083; Type ID: n=1,054; Spatial: n=144

**Sample low-agreement questions** illustrate genuine image ambiguity:
1. Counting bottles inside a bag (annotators chose c/c/d/a/b — max 2/5)
2. Recycling material classification for a lip balm case (a/a/c/b/b — split 2-2-1)
3. Counting cardboard boxes visible in image (d/b/d/b/c — split 2-2-1)

These are not annotation errors — they reflect genuine visual ambiguity in the images.

**Figure**: `.omc/scientist/figures/03_difficulty_distribution.png`

[LIMITATION] "Label noise estimate" is a theoretical upper bound based on disagreement rate, not validated against a ground-truth label. True error rate may be lower if one annotator systematically outperforms others (but pairwise analysis shows no such pattern). Question type classification uses keyword heuristics and may misclassify some questions.

---

## [FINDING:F6] Pseudo-Label Strategy

**Hypothesis**: A threshold of >=3/5 agreement maximizes training data quality vs. quantity tradeoff.

**Result**: Supported with caveats. Best strategy depends on training objective.

**Threshold analysis** (5-annotator rows only, n=4,301):

| Strategy | Threshold | Rows available | Est. label noise | Effective rows |
|----------|-----------|---------------|-----------------|----------------|
| A: High only | >=4/5 | 891 | 20.0% | ~713 |
| C: Moderate | >=3/5 | 3,430 | 34.8% | ~2,238 |
| B: Weighted all | (all, weighted) | 4,301 | 39.9% | ~2,584 (sum of weights) |
| D: Model re-label | >=3/5 + model on ties | ~4,301 | ~25% (est.) | ~3,226 |

[STAT:n] n=4,301 fully-annotated rows; train.csv has 5,073 rows

**Recommended strategies (in order of preference)**:

**1. Strategy D (Best): Model-Ensemble Re-labeling**
- Use high-confidence (>=4/5, n=891) + moderate (3/5, n=2,539) as pseudo-labeled training additions  
- For the 871 tied rows (2/2/1 or 2/1/1/1), run a baseline model prediction to break ties  
- Estimated usable data: ~4,301 rows with ~25% theoretical noise  
- Combined with train.csv: ~9,374 rows total  

**2. Strategy C (Practical): Moderate threshold (>=3/5)**
- 3,430 dev rows + 5,073 train rows = **8,503 total**  
- 17% increase in training data with manageable noise (~35%)  
- No model re-labeling required  

**3. Strategy B (Maximal): Weighted training (all rows)**
- All 4,301 dev rows used, with sample_weight = agreement_ratio (0.4–0.8)  
- Maximizes coverage; effective sample ≈ 2,584 rows  
- Suitable for models that support instance weighting (e.g., XGBoost, weighted CrossEntropy)  

**4. Strategy A (Conservative): High confidence only**
- Only 891 rows — smallest gain (~17.6% increase over train-only)  
- Lowest noise (~20%), but noise is still significant  
- Recommended only if model is highly sensitive to label noise  

**Critical observations**:
1. With kappa=0.16, even "high confidence" labels have 20% expected noise — this is above typical thresholds for clean pseudo-labels (usually <10%)
2. The question is inherently ambiguous: even human experts disagree. Model performance upper bound may be limited by annotation noise
3. Weighted training (Strategy B) likely outperforms hard-threshold strategies because it preserves all information without arbitrary cutoffs

[STAT:effect_size] Fleiss' kappa = 0.164 (slight agreement); pairwise mean = 0.373
[STAT:n] n = 4,301 (fully annotated) + 112 (partial) = 4,413 total dev rows
[STAT:ci] No formal CI computed — annotation noise is estimated from agreement ratios, not from a held-out test

**Figure**: `.omc/scientist/figures/04_pseudo_label_strategy.png`

[LIMITATION]
1. No ground-truth "gold label" exists for dev.csv — all noise estimates are theoretical based on agreement ratios, not empirically validated
2. The Fleiss' kappa calculation assumes exchangeability of annotators — if annotator 3 is systematically more expert, weighting by annotator quality could improve pseudo-labels
3. Missing annotations (112/4,413 rows, ~2.5%) are excluded from full analysis but may not be MCAR (missing completely at random) — they might be the hardest questions
4. Using dev data for training risks distribution shift if dev and test images differ in style/difficulty
5. Answer position bias (d=38% in high-agreement rows) may introduce spurious correlations if not addressed in the model prompt or answer randomization

---

## Summary Table

| Finding | Key Statistic | Implication |
|---------|--------------|-------------|
| F4: IAA is very low | Fleiss' κ = 0.164 (slight) | Dev labels are noisy; don't treat as gold |
| F5: No unanimous agreement | 0% unanimous, 20.7% strong (4/5) | Always use majority vote, not any single annotator |
| F5: 59% weak majority | 3/5 agree on 2,539 rows | Large pool available but high noise (~40%) |
| F6: Best strategy | Threshold >=3/5 OR weighted all | +3,430 rows; noise ~35% |

**Figures generated**:
- `01_agreement_levels.png` — bar chart of agreement distribution + strategy comparison
- `02_annotator_comparison.png` — pairwise annotator agreement heatmap
- `03_difficulty_distribution.png` — agreement by question type + pie chart
- `04_pseudo_label_strategy.png` — strategy comparison (total vs effective rows)
