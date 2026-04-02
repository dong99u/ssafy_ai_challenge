# VQA Question Type Classification & Distribution Analysis
Generated: 2026-04-03T01:26:58.746568

## [OBJECTIVE]
Classify VQA dataset questions (Korean recycling/waste domain) into semantic types,
analyze their distribution, answer balance, and difficulty — to inform model strategy.

---

## [DATA]
- **Train**: 5,073 rows, 8 columns (id, path, question, a, b, c, d, answer). No missing values.
- **Dev**: 4,413 rows, 12 columns (+ answer1–answer5 from 5 annotators). 2–36 missing annotator answers.
- **Task**: 4-choice multiple choice VQA about recycling/waste items in Korean.

---

## [FINDING 1] Question Types: 3 dominant types cover 84% of training data

Exclusive primary type classification (priority-ordered):

| Type | Train N | Train % | Dev N | Dev % |
|------|---------|---------|-------|-------|
| COUNT | 1,746 | 34.4% | 3,151 | 71.4% |
| MATERIAL | 1,591 | 31.4% | 555 | 12.6% |
| OBJECT_ID | 952 | 18.8% | 339 | 7.7% |
| COLOR | 322 | 6.3% | 103 | 2.3% |
| RECYCLE_CLASS | 230 | 4.5% | 58 | 1.3% |
| LOCATION | 125 | 2.5% | 150 | 3.4% |
| MATERIAL_TYPE_ID | 57 | 1.1% | 33 | 0.7% |
| BRAND_PRODUCT | 30 | 0.6% | 6 | 0.1% |
| OTHER | 13 | 0.3% | 13 | 0.3% |
| CONDITION | 7 | 0.1% | 5 | 0.1% |

[STAT:n] Train n=5,073 / Dev n=4,413
[STAT:effect_size] COUNT+MATERIAL+OBJECT_ID = 84.6% of train, 91.7% of dev

---

## [FINDING 2] MASSIVE train/dev distribution shift — COUNT dominates dev at 71.4%

The dev set has COUNT questions at 71.4% vs 34.4% in train (+37 percentage points).
MATERIAL drops from 31.4% to 12.6% (-18.8pp), OBJECT_ID from 18.8% to 7.7% (-11.1pp).

[STAT:effect_size] Δ(COUNT) = +37.0pp — largest shift of any type
[STAT:n] Dev COUNT: 3,151 questions; 1,187 unique question texts (many repeated across images)
[STAT:effect_size] Top repeated dev question appears 139 times across different images

**Implication**: A model evaluated on dev needs to be particularly strong at counting/enumerating
recyclable objects. This is likely the highest-ROI capability to optimize.

---

## [FINDING 3] Answer choices are uniformly distributed across all question types

No question type shows statistically significant deviation from the expected 25% per answer (a/b/c/d).
Chi-square tests vs uniform H₀:

| Type | chi² | p-value | Interpretation |
|------|------|---------|---------------|
| COUNT | 2.08 | 0.556 | Uniform ✓ |
| MATERIAL | 4.98 | 0.173 | Uniform ✓ |
| OBJECT_ID | 4.22 | 0.239 | Uniform ✓ |
| COLOR | 3.69 | 0.297 | Uniform ✓ |
| RECYCLE_CLASS | 4.12 | 0.249 | Uniform ✓ |
| LOCATION | 0.41 | 0.939 | Uniform ✓ |

[STAT:p_value] All p > 0.05, no type shows answer bias
[STAT:n] Largest group: COUNT n=1,746

**Implication**: No positional answer shortcut (e.g., "always choose C") exists per type.
The dataset is well-balanced — models must truly understand the visual content.

---

## [FINDING 4] Annotator agreement is moderate across all types, never reaching full consensus

Dev set (5 annotators per question) agreement distribution:
- 2/5 agreement (hardest): 903 questions (20.5%)
- 3/5 agreement (medium): 2,606 questions (59.1%)
- 4/5 agreement (easiest): 902 questions (20.4%)
- No question achieves 5/5 full agreement

Difficulty by type (hard = ≤2/5 agree):

| Type | Hard (≤2/5) | Medium (3/5) | Easy (≥4/5) | Mean |
|------|------------|--------------|-------------|------|
| LOCATION | 24.7% | 60.0% | 15.3% | 2.91 |
| COUNT | 22.8% | 54.5% | 22.7% | 3.00 |
| MATERIAL_TYPE_ID | 21.2% | 57.6% | 21.2% | 3.00 |
| OBJECT_ID | 15.9% | 69.9% | 14.2% | 2.98 |
| MATERIAL | 12.8% | 73.9% | 13.3% | 3.01 |
| COLOR | 9.7% | 69.9% | 20.4% | 3.11 |
| RECYCLE_CLASS | 12.1% | 70.7% | 17.2% | 3.05 |

[STAT:n] Dev n=4,413
[STAT:effect_size] LOCATION most ambiguous (24.7% hard); COLOR least ambiguous (9.7% hard)

**Implication**: LOCATION questions are hardest for humans — spatial reasoning about where objects
sit in the image is ambiguous. COLOR questions are clearest. COUNT is both the largest category
AND has ~23% hard cases — critical to get right.

---

## [FINDING 5] Answer option structure is highly type-specific and semantically constrained

- COUNT options: 94.4% use "N개" (number+개) format; numbers 1–4 dominate (82% of occurrences)
- MATERIAL options: 83.6% explicitly name material keywords (종이/플라스틱/유리/금속)
- COLOR options: 97.1% contain color keywords; 파란/노란/초록/빨간 most common
- RECYCLE_CLASS options: 98.9% use material classification keywords

[STAT:n] COUNT option analysis: 6,984 answer options analyzed
[STAT:effect_size] Numbers 1-4 cover ~82% of count answers — most objects in images are small sets

**Implication**: Type-specific post-processing or chain-of-thought prompting can leverage
the predictable option structure to validate model outputs before final answer selection.

---

## [FINDING 6] High question template repetition — same questions applied to many images

- Train: 5,073 total, only 2,303 unique questions (45.4% unique rate)
- Top template: "사진에 보이는 재활용품 중 플라스틱 재질인 것은 무엇인가요?" appears 245 times
- Dev: top COUNT template repeated 139 times across different images

[STAT:n] 2,770 duplicate question instances in train (54.6% repeat rate)

**Implication**: The model cannot memorize question-answer pairs; it must rely on image understanding.
However, answer option sets for recurring templates are worth analyzing for systematic errors.

---

## [LIMITATION]
1. **Type classification is keyword-based**: multi-type questions (79.6% of train) are reduced to one
   primary type by priority ordering. Some nuanced questions may be miscategorized.
2. **Dev distribution shift is severe**: evaluation on dev heavily weights COUNT (71%), making dev
   scores a poor proxy for general VQA ability if train distribution is more representative.
3. **Agreement as difficulty proxy**: annotator disagreement (≤2/5) may reflect annotation variance
   as much as genuine question ambiguity. True difficulty requires model performance data.
4. **No image features analyzed**: this analysis is text-only; image complexity (clutter, occlusion,
   lighting) may interact with question type difficulty in ways not captured here.
5. **Small type samples**: CONDITION (n=7), BRAND_PRODUCT (n=30), OTHER (n=13) — insufficient for
   statistical conclusions; chi-square and agreement stats for these types are unreliable.

---

## Strategic Recommendations

1. **Prioritize COUNT accuracy**: 71% of dev is COUNT. Prompts should explicitly guide counting
   (e.g., chain-of-thought: "First count each recycling type visible, then select the answer").
2. **Answer option filtering**: Since COUNT answers are mostly 1–4 and MATERIAL/RECYCLE_CLASS use
   a closed vocabulary, use option-aware prompting.
3. **Treat dev as COUNT-heavy benchmark**: models optimized for COUNT will score higher on dev;
   balanced capability requires ensuring MATERIAL/OBJECT_ID don't regress.
4. **COLOR is the "free" category**: 9.7% hard rate, closed answer set — ensure high accuracy here.
5. **LOCATION questions need spatial reasoning**: 24.7% hard rate, likely needs explicit spatial
   description in the chain-of-thought.

---

## Files
- Report: `.omc/scientist/reports/20260403_012658_question_type_analysis.md`
- Figure 1 (distribution): `.omc/scientist/figures/question_type_distribution.png`
- Figure 2 (agreement): `.omc/scientist/figures/dev_agreement_analysis.png`
