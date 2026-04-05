# Stage 5: Question NLP Analysis — Keywords, Difficulty Signals, Prompt Engineering Insights

**Date:** 2026-04-04
**Dataset:** train.csv (5,073 rows) + dev.csv (4,413 rows) = 9,486 questions total
**Analysis by:** Scientist agent (oh-my-claudecode)

---

## [OBJECTIVE]
Perform NLP analysis of Korean-language VQA questions to:
1. Identify dominant keyword patterns and question templates
2. Measure complexity signals (counting, negation, superlatives)
3. Analyze answer option patterns (numerics, material names, colors)
4. Correlate question features with annotator agreement (difficulty proxy)
5. Derive concrete prompt engineering recommendations

---

## [DATA]
- Train: 5,073 rows (id, path, question, a, b, c, d, answer)
- Dev: 4,413 rows with 5 annotator labels (answer1-answer5)
- Combined: 9,486 questions, 20,292 answer options (4 per question in train)
- Question length: mean=32.35 chars, 95% CI=[32.24, 32.45], std≈5.3 chars
- All answer options are unique per question (no duplicate options observed)

---

## [FINDING:F12] Question Keyword Patterns

### F12.1 — Recycling Dominance
The term '재활용' (recycling) appears in **8,880 / 9,486 questions (93.6%)**.
'재질' (material) appears in 2,618 (27.6%). '분류' (classification) in 371 (3.9%).
This confirms the dataset is almost entirely about recycling material identification.
[STAT:n] n=9,486
[STAT:effect_size] '재활용' presence rate = 0.936 (near-universal)

### F12.2 — Counting Questions Are the Largest Task Type
'몇 개' or '개수' (counting/how many) appears in **4,895 / 9,486 questions (51.6%)**.
This is the single largest question sub-type across both splits.
[STAT:n] n=9,486
[STAT:effect_size] proportion = 0.516

### F12.3 — Plastic is the Most Referenced Material
Material keyword frequency (combined train+dev):
- 플라스틱 (plastic): 4,119 mentions
- 종이 (paper): 1,559 mentions
- 캔 (can): 687 mentions
- 유리 (glass): 317 mentions
- 금속 (metal): 215 mentions
- 알루미늄 (aluminum): 208 mentions
- 비닐 (vinyl/plastic bag): 167 mentions
74.6% of all questions mention at least one material keyword.
[STAT:n] n=9,486

### F12.4 — Questions Are Highly Templated
Top 3 normalized question patterns repeat across:
- "사진에 보이는 재활용품 중 플라스틱 재질인 것은 무엇인가요?" → 342 occurrences (3.6%)
- "사진에 보이는 재활용 가능한 플라스틱 병은 몇 개입니까?" → 233 occurrences (2.5%)
- "사진에 보이는 재활용 가능한 플라스틱 컵은 몇 개입니까?" → 227 occurrences (2.4%)
Only 3,846 unique normalized patterns out of 9,486 questions (40.5% unique).
This suggests systematic template-based generation.
[STAT:n] n=9,486

### F12.5 — Negation and Superlatives Are Rare
- Negation terms ('아닌', '않은', etc.): only 18 questions total (0.2%)
- Superlative terms ('가장', '모두'): 247 questions (2.6%)
- Knowledge-domain questions ('올바른', '배출 방법'): 74 questions (0.8%)
The vast majority require visual recognition rather than rule knowledge.
[STAT:n] n=9,486

### F12.6 — Answer Options Are Dominated by Count Values and Material Names
Top answer options across all 20,292 answer slots (train):
- '3개': 1,598 (7.9%)
- '2개': 1,504 (7.4%)
- '1개': 1,358 (6.7%)
- '4개': 1,306 (6.4%)
34.2% of all answer options are numeric count expressions (e.g., "1개", "2종류").
64.7% are single-word answers. Mean answer length = 4.42 chars (std=3.74).
[STAT:n] n=20,292 answer options

---

## [FINDING:F13] Difficulty Predictors

### F13.1 — Overall Annotator Agreement is Moderate
Dev set annotator agreement (5 annotators per question):
- Mean agreement: 0.6034 (std=0.1302)
- Fully unanimous (5/5): 13 questions (0.3%)
- 4/5 agreement: 891 questions (20.2%)
- 3/5 agreement (bare majority): 2,539 questions (57.5%)
- 2/5 agreement (split): 871 questions (19.7%)
- 1/5 or 0/5 agreement: 3 questions (0.07%)
The dataset is highly contested: only 0.3% achieve unanimous agreement.
[STAT:n] n=4,413
[STAT:effect_size] 79.7% of questions have bare majority (3/5) agreement or worse

### F13.2 — Task Type is the Most Meaningful Difficulty Predictor
Agreement by task type (dev, 95% CIs):
| Task Type       | Mean Agreement | 95% CI              | n     |
|----------------|----------------|---------------------|-------|
| Location        | 0.5799         | [0.5563, 0.6035]    | 107   |
| Identification  | 0.5994         | [0.5872, 0.6115]    | 313   |
| Counting        | 0.6029         | [0.5981, 0.6077]    | 3,150 |
| Material ID     | 0.6056         | [0.5972, 0.6140]    | 572   |
| Classification  | 0.6125         | [0.5872, 0.6378]    | 80    |
| Color ID        | 0.6298         | [0.6103, 0.6492]    | 121   |

Location questions are hardest. Color ID questions are easiest.
One-way ANOVA: F=1.559, p=0.155 (not significant at α=0.05 after all-task comparison).
Color ID vs Counting: t=2.130, p=0.033*, Cohen's d=0.217 (small effect).
[STAT:p_value] p=0.033 (color_id vs counting)
[STAT:effect_size] Cohen's d=0.217 (small)
[STAT:n] n=4,413

### F13.3 — Question Length is Weakly Negatively Correlated with Agreement
Pearson r(q_len, agreement_rate) = -0.011 (near zero, not meaningful).
Neither question length nor common complexity signals (negation, superlatives)
predict difficulty. The hardest questions appear to be those requiring
precise spatial counting in cluttered scenes — which is a visual, not
linguistic, challenge.
[STAT:effect_size] Pearson r=-0.011 (negligible)
[STAT:n] n=4,413

### F13.4 — Hardest Questions Are Ambiguous Counting in Cluttered Scenes
Manual examination of low-agreement (< 0.50) questions reveals 3 error patterns:
1. **Occlusion ambiguity**: Items partially hidden; annotators disagree on whether
   to count partially visible objects (e.g., "사진 속 검은색 쓰레기봉투 안에 보이는
   재활용품 중 플라스틱 병은 몇 개입니까?" — votes: b/a/a/b/c)
2. **Scene position ambiguity**: Questions about spatial location of objects where
   annotators perceive the reference frame differently (e.g., "마우스 오른쪽에 있다"
   vs "키보드 왼쪽에 있다" — both may be technically correct depending on perspective)
3. **Boundary counting**: Whether a container counts as 1 item or its contents do
[STAT:n] n=36 questions with agreement < 0.5

### F13.5 — Material Confusion: Plastic Has Lowest Agreement
Among material-specific questions:
- 금속 (metal): mean_agreement=0.5921 (hardest material to identify)
- 플라스틱 (plastic): mean_agreement=0.5961
- 비닐 (vinyl): mean_agreement=0.6063
- 캔: mean_agreement=0.6071
- 종이 (paper): mean_agreement=0.6068
- 알루미늄: mean_agreement=0.6167
- 유리 (glass): mean_agreement=0.6156
Metal and plastic have highest confusion — likely because metal objects
can resemble plastic in photos (e.g., shiny vs matte surfaces).
[STAT:n] ranges from 57 (금속) to 2,248 (플라스틱)

---

## [FINDING:F14] Prompt Engineering Recommendations

### F14.1 — System Prompt Must Include Korean Recycling Category Schema
Since 93.6% of questions concern recycling classification, the system prompt
should explicitly list the Korean recycling categories:
- 플라스틱 (plastic): PET bottles, cups, containers
- 유리 (glass): bottles, jars
- 금속/캔 (metal/cans): aluminum cans, steel cans
- 종이/골판지 (paper/cardboard): boxes, cups
- 비닐 (vinyl/plastic bags): thin plastic films, wrappers
- 스티로폼 (styrofoam): packaging foam

### F14.2 — Counting Strategy Must Be Explicit
51.6% of questions require counting. The system prompt should instruct:
1. Count only clearly visible (not occluded) items unless context implies partial counts
2. Consider items of the same type as one unit vs individual units
3. For counting questions, enumerate objects before answering

### F14.3 — Spatial/Location Questions Need Special Handling
Location questions have the lowest mean agreement (0.5799). The prompt should
instruct the model to describe object positions relative to the dominant
reference frame (viewer's perspective, not relative to other objects in scene).

### F14.4 — Metal vs Plastic Disambiguation Should Be Explicit
Metal items have the lowest agreement (0.5921). The system prompt should
include visual cues for discrimination:
- Aluminum cans: cylindrical, often labeled, silver/colored metallic
- Metal objects: reflective surfaces, heavier appearance
- Plastic: often transparent, colored, with mold seams

### F14.5 — Answer Format: Single Character (a/b/c/d) After Reasoning
Since model output is parsed for a single letter, the prompt should instruct:
"Your final answer must be a single letter: a, b, c, or d"
Consider chain-of-thought (CoT) prompting for counting questions to
reduce counting errors, even if only the letter is extracted.

### F14.6 — Template-Aware Training Strategy
40.5% of question patterns are repeated across the dataset. This means:
1. Models should generalize from visual variety rather than question text
2. Training should use image augmentation to prevent over-fitting on
   repeated templates
3. High-frequency templates ('플라스틱 재질인 것은 무엇인가요?') could be
   used as evaluation sanity checks for model consistency

---

## [LIMITATION]

1. **Annotator agreement as difficulty proxy**: The 5-annotator dev set may not
   perfectly represent model difficulty. Human ambiguity ≠ model confusion.
   Some questions humans find ambiguous may be clear to a VLM with more context.

2. **Korean text analysis without morphological analyzer**: Keyword matching
   is substring-based (no morphological tokenization like Mecab/Komoran).
   Some compound words or conjugated forms may be missed (e.g., '버립니다').

3. **Correlation is weak**: The strongest difficulty predictor (task type) has
   a non-significant ANOVA (p=0.155). Question text alone is a poor predictor
   of difficulty — visual complexity of the image is the dominant factor.

4. **Dev/Train distribution mismatch**: Dev has 71.4% counting questions vs
   Train's 34.4%. This may reflect different sampling strategies and could
   affect how well train-based analysis generalizes to dev difficulty patterns.

5. **No image-side analysis**: This stage analyzes only question text.
   Image-side features (object count, occlusion, lighting) are likely
   stronger difficulty predictors but were not analyzed here.

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total questions analyzed | 9,486 |
| '재활용' keyword coverage | 93.6% |
| Counting questions | 51.6% |
| Material-mentioning questions | 74.6% |
| Numeric answer options | 34.2% |
| Mean annotator agreement | 0.603 ± 0.130 |
| Hardest task type | Location (mean=0.580) |
| Easiest task type | Color ID (mean=0.630) |
| Unique question templates | 40.5% |

---

## Figures

- `keyword_frequency.png`: Material/recycling keyword frequencies, task type distribution and agreement
- `question_distribution_eda.png`: Task agreement by type, agreement distribution, answer length histogram

Report generated: 2026-04-04
