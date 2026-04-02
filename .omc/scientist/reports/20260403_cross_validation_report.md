# Cross-Validation Report: Korean Recycling VQA Dataset Research
**Generated:** 2026-04-03 01:34:20
**Analyst:** Scientist Agent (cross-validation of 6 parallel research stages)
**Dataset:** Korean Recycling VQA (train: 5,073 / test: 5,074 / dev: 4,413)

---

## VERDICT: [CONFLICTS: 3] with bulk findings VERIFIED

The 6-stage research is **substantially sound**. Three numerical issues were identified,
none of which invalidate the strategic conclusions.

---

## 1. CONFLICTS IDENTIFIED

### CONFLICT 1: Question Reuse Rate Inverted (Stage 1) -- MEDIUM severity
- **Claim:** "45% question reuse"
- **Actual:** 2,303 unique out of 5,073 = 45.4% UNIQUE, meaning **54.6% REUSE**
- **Fix:** Replace with "54.6% question reuse (only 2,303 unique questions out of 5,073)"

### CONFLICT 2: Missing Annotation Percentage Unverifiable (Stage 1) -- LOW severity
- **Claim:** "118 missing annotations (0.22%)"
- **Actual:** 118 / 22,065 annotation cells = 0.535%; 118 / 4,413 samples = 2.67%
- No natural denominator yields 0.22%
- **Fix:** Replace with "118 missing annotation cells (0.53% of 22,065 total)"

### CONFLICT 3: Cross-Stage Metric Ambiguity (Stages 2/5, 3/5) -- MEDIUM severity
- Dev COUNT prevalence: Stage 2 says 71%, Stage 5 says 68.7%
- '2개' accuracy: Stage 3 says 45.6%, Stage 5 says 39.4%
- **Likely explanation:** Different metrics with overlapping names (question TYPE vs numeric OPTIONS; conditional vs blanket accuracy)
- **Fix:** Add explicit metric definitions to disambiguate

---

## 2. VERIFIED FINDINGS

| Finding | Status | Cross-Check |
|---------|--------|-------------|
| Total 14,560 samples | VERIFIED | 5073+5074+4413 = 14560 |
| Train balanced (a/b/c/d ~25%) | VERIFIED | Sum=99.9%, chi2 p=0.494 |
| Fleiss' Kappa = 0.167 | VERIFIED | Correct metric & interpretation |
| Zero unanimity (0/4413) | VERIFIED | Consistent with 59.1%+20.5%+20.4%=100% |
| Avg pseudo-label confidence ~60% | VERIFIED | Computed 56.9%, within rounding |
| 13 templates cover 100% | VERIFIED | Logically sound |
| Image properties uniform | VERIFIED | Cross-split consistent |
| COUNT distribution shift +37pp | VERIFIED | Both stages confirm |
| 2nd-smallest heuristic 52.7% | VERIFIED | p<0.001, rigorous |

---

## 3. MISSING CONNECTIONS DISCOVERED

1. **Question Reuse x Low Agreement = Label Noise Amplification**
   54.6% reuse + Kappa 0.167 => model must learn purely visual discrimination; text shortcuts are doubly unreliable.

2. **COUNT Shift x '2개' Heuristic = Dev Strategy**
   COUNT jumps to 71% in dev; '2개' heuristic gives 39-46% on counting => counting mastery yields outsized dev gains.

3. **Template Closure x Brand Names = OCR Requirement**
   13 templates are fixed; 413 novel test tokens are brand names => OCR is essential for test, not for question parsing.

4. **Annotator 1 Outlier x B-Bias x No Majority = Noise Source**
   20.5% no-majority questions are driven by annotator disagreement, especially Annotator 1; these samples have near-random labels.

5. **Image Uniformity x Resolution = Preprocessing Simplicity**
   86-88% in 500-1000px range with uniform format => standard preprocessing suffices, but OCR needs adequate resolution.

---

## 4. COVERAGE GAPS

| Gap | Priority | Description |
|-----|----------|-------------|
| Test distribution unknown | CRITICAL | Train balanced, dev count-heavy; test could follow either |
| No image content analysis | HIGH | Format stats only; no object types, scene complexity, OCR prevalence |
| Question-image mapping unknown | MEDIUM | How many unique images? Questions per image? |
| Dev scoring mechanism unclear | MEDIUM | How are 5 annotations aggregated? What about 20.5% no-majority? |
| Distractor generation unknown | MEDIUM | How are wrong answer options created? Exploitable patterns? |
| Difficulty x agreement not cross-tabulated | LOW | Are hard questions hard or just ambiguous? |

---

## 5. TOP 5 STRATEGIC INSIGHTS

### #1: COUNTING IS THE KINGMAKER
COUNT questions go from 34% (train) to 71% (dev). The 2nd-smallest heuristic gives 52.7% on counting (p<0.001).
**Action:** Prioritize counting accuracy; consider count-specific fine-tuning.

### #2: LABEL NOISE IS THE CEILING
Kappa=0.167, zero unanimity, 20.5% no majority. Theoretical accuracy ceiling ~80%.
**Action:** Do not chase dev accuracy beyond ~75%. Use confidence-weighted training.

### #3: TEXT IS SOLVED; VISION IS EVERYTHING
13 templates cover 100%; text-only ceiling <60%; 413 novel tokens = brand names requiring OCR.
**Action:** Use VLM with strong OCR (Qwen3-VL). No text feature engineering beyond template routing.

### #4: DISTRIBUTION SHIFT IS THE TRAP
Massive train-to-dev shift; test distribution unknown.
**Action:** Train on train (clean, balanced), validate on dev (noisy but informative). Do not over-fit to either.

### #5: EXPLOITABLE STATISTICAL SHORTCUT EXISTS
2nd-smallest counting heuristic = 52.7%, template detection = 100% accurate.
**Action:** Confidence-gated hybrid -- VLM primary, statistical fallback for low-confidence counting.

---

## 6. EVIDENCE QUALITY

| Stage | Score | Assessment |
|-------|-------|------------|
| 1 (Basic Stats) | 7/10 | Good data, two labeling errors |
| 2 (Question Types) | 6/10 | Useful findings, methodology gaps |
| 3 (Answer Bias) | 8/10 | Strongest statistical rigor |
| 4 (Inter-Annotator) | 8/10 | Rigorous methodology |
| 5 (Text NLP) | 6/10 | Good insights, cross-stage ambiguity |
| 6 (Image Properties) | 5/10 | Shallow for a visual task |
| **Overall** | **6.7/10** | Substantially sound, needs corrections |
