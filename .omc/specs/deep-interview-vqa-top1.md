# Deep Interview Spec: SSAFY VQA Competition — 0.93+ Accuracy in 24h

## Metadata
- Interview ID: vqa-competition-top1-24h
- Rounds: 8
- Final Ambiguity Score: 9.5%
- Type: brownfield
- Generated: 2026-04-04
- Threshold: 20%
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.95 | 35% | 0.333 |
| Constraint Clarity | 0.92 | 25% | 0.230 |
| Success Criteria | 0.82 | 25% | 0.205 |
| Context Clarity | 0.92 | 15% | 0.138 |
| **Total Clarity** | | | **0.905** |
| **Ambiguity** | | | **9.5%** |

## Goal
Improve SSAFY Recycling VQA competition accuracy from **0.87 → 0.93+** (ideally approaching 0.95782 for #1) within 24 hours using a single H100 80GB GPU, working alone.

## Current State
- **Best submission:** 0.87 accuracy
- **Model:** Qwen2.5-VL-7B-Instruct, bf16 (no quantization)
- **LoRA:** r=16, alpha=32, dropout=0.05, target_modules=[q/k/v/o_proj, gate/up/down_proj]
- **Training:** 2 epochs, LR=2e-5, batch_size=1, grad_accum=8, AdamW, linear warmup (5%)
- **Data:** train.csv only (5073 samples, 90/10 sequential split)
- **Image:** 384×384 pixels
- **Prompt:** Generic "You are a helpful visual question answering assistant"
- **Inference:** text generation (max_new_tokens=2) + regex parsing, defaults to "a" on failure
- **Failed attempt:** Qwen3.5-9B → 0.67 (external data noise + broken ensemble)

## Improvement Vectors (Priority Order)

### Tier 1: High Impact, Low Risk (~+0.04-0.06)
1. **Add dev.csv with majority-vote labels** — 4413 additional samples with 5 annotator labels. Take majority vote as pseudo-gold. Combined training: ~9,486 samples (nearly 2× data).
2. **Increase image resolution** — 384→672px (or 1024px if memory allows). Recycling items have fine details (labels, textures, material types) lost at 384.
3. **Logprob-based answer selection** — Instead of generating text and parsing, compute logprobs for tokens "a","b","c","d" and pick highest. Eliminates parsing failures entirely.

### Tier 2: Medium Impact (~+0.02-0.03)
4. **Domain-specific system prompt** — Recycling/waste classification expertise prompt in Korean.
5. **LoRA r=32, alpha=64** — More adapter capacity on H100.
6. **3 epochs with cosine schedule** — More training + smoother LR decay.
7. **Stratified split** — Split by question type instead of sequential.
8. **Learning rate 1e-5** — More conservative LR for larger dataset.

### Tier 3: Bonus if Time Permits (~+0.01-0.02)
9. **Ensemble 2 models** — Different seeds/LR/LoRA configs, majority vote.
10. **Train+dev combined (no validation holdout)** — For final submission, train on ALL data.
11. **Test-Time Augmentation** — Horizontal flip at inference.

## Constraints
- 1× H100 80GB GPU
- 24 hours total (training + inference + submission)
- Working alone (no parallel team members)
- HuggingFace pretrained models only; no API calls
- Daily submission limit: 20 (plenty for 24h)
- Reproducibility required

## Non-Goals
- Trying Qwen3.5-9B again (too slow, already failed)
- External data augmentation (caused 0.67 failure)
- Full fine-tuning (LoRA is sufficient and faster)
- Qwen2.5-VL-72B (training time too long for 24h solo)
- Complex multi-model ensemble (>2 models)

## Acceptance Criteria
- [ ] Public leaderboard accuracy ≥ 0.93
- [ ] Training completes within ~10 hours (leaving time for iteration)
- [ ] dev.csv majority-vote pseudo-labels incorporated into training
- [ ] Logprob-based inference implemented (no text parsing failures)
- [ ] At least 2 submissions made (single model + optimized variant)
- [ ] Error analysis performed on validation set before final submission

## 24-Hour Execution Plan

| Phase | Hours | Action | Expected Result |
|-------|-------|--------|----------------|
| 1 | 0-1h | **Data prep**: dev.csv majority vote, combine train+dev, stratified split | ~9.5k training samples ready |
| 2 | 1-2h | **Notebook prep**: domain prompt, logprob inference, higher resolution, LoRA r=32 | Optimized notebook ready |
| 3 | 2-7h | **Train Model A**: Qwen2.5-VL-7B, r=32, 672px, 3 epochs, train+dev, cosine schedule | Best single model (~0.91-0.93) |
| 4 | 7-8h | **Inference + Submit**: Logprob inference on test set, submit | First improved submission |
| 5 | 8-9h | **Error analysis**: Analyze validation errors by question type | Know weak areas |
| 6 | 9-14h | **Train Model B**: Different seed (123), LR=5e-5, LoRA r=64 | Ensemble candidate |
| 7 | 14-16h | **Ensemble + Submit**: Majority vote of A+B, submit both individual + ensemble | Best combined score |
| 8 | 16-20h | **Iterate**: Based on error analysis — retrain with fixes or try InternVL2-8B | Push toward 0.95+ |
| 9 | 20-24h | **Final push**: Train on ALL data (no validation holdout), submit final | Maximum score submission |

## Key Technical Details

### dev.csv Majority Vote
```python
# dev.csv has answer1-answer5 columns
from collections import Counter
def majority_vote(row):
    votes = [row[f'answer{i}'] for i in range(1, 6)]
    return Counter(votes).most_common(1)[0][0]
dev_df['answer'] = dev_df.apply(majority_vote, axis=1)
```

### Logprob Inference
```python
# Instead of model.generate(), use forward pass + logprobs
choice_tokens = [processor.tokenizer.encode(c, add_special_tokens=False)[0] for c in ['a','b','c','d']]
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]  # last token logits
    choice_logits = logits[:, choice_tokens]
    pred_idx = choice_logits.argmax(dim=-1)
    pred = ['a','b','c','d'][pred_idx.item()]
```

### Domain-Specific Prompt
```python
SYSTEM_INSTRUCT = (
    "당신은 재활용 및 폐기물 분류 전문가입니다. "
    "이미지를 보고 재활용 관련 질문에 정확하게 답변하세요. "
    "반드시 a, b, c, d 중 하나의 소문자 한 글자로만 답하세요."
)
```

### Image Resolution
```python
IMAGE_SIZE = 672  # or 1024 if H100 memory allows with batch_size=1
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=IMAGE_SIZE * IMAGE_SIZE,
    max_pixels=IMAGE_SIZE * IMAGE_SIZE,
)
```

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| Bigger model = better | Qwen3.5-9B scored 0.67 | Model size isn't the bottleneck; data quality and training setup are |
| External data helps | External data caused 0.67 | Stick to competition data only; dev.csv is the goldmine |
| More GPU time needed | Only 2 epochs trained | 3 epochs + dev data + higher resolution can close most of the gap |
| Text generation is fine | extract_choice defaults to "a" | Logprob-based selection eliminates parsing failures |
| Generic prompt is OK | No domain knowledge in prompt | Recycling-specific Korean prompt should help |
| Sequential split is OK | No stratification | Stratified by question type prevents validation bias |

## Ontology (Key Entities)

| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| Qwen2.5-VL-7B | core model | bf16, flash_attn_2, gradient_ckpt | fine-tuned via LoRA adapter |
| LoRA Adapter | training method | r, alpha, dropout, target_modules | applied to base model |
| train.csv | training data | 5073 rows, id/path/question/a-d/answer | primary training source |
| dev.csv | training data | 4413 rows, 5 annotator labels | majority-vote → pseudo-gold |
| test.csv | inference data | 5074 rows, no answers | prediction target |
| H100 GPU | constraint | 80GB VRAM, single GPU, 24h | limits batch size, model size |
| Submission | output | CSV with id + answer | uploaded to leaderboard |
| System Prompt | config | SYSTEM_INSTRUCT string | shapes model response style |

## Interview Transcript
<details>
<summary>Full Q&A (8 rounds)</summary>

### Round 1
**Q:** 지금 사용할 수 있는 GPU 환경이 무엇인가요? 그리고 현재까지 제출한 최고 점수는 얼마인가요?
**A:** H100 / A100 80GB 서버
**Ambiguity:** ~80% (Goal: 0.5, Constraints: 0.6, Criteria: 0.3)

### Round 2
**Q:** 현재까지 제출한 최고 점수(accuracy)는 얼마인가요?
**A:** 0.95782 (→ corrected in Round 3)
**Ambiguity:** ~60%

### Round 3
**Q:** 현재 리더보드 상황이 어떻게 되나요?
**A:** 잘못 말했어. 우리 팀 최고 점수는 0.87이고, 전체 1위가 0.95782임.
**Ambiguity:** ~40% (Goal: 0.85, Constraints: 0.70, Criteria: 0.50)

### Round 4 (Contrarian prep)
**Q:** 0.87 점수를 낸 모델/설정이 뭔였나요? GPU 병렬 가능?
**A:** Qwen2.5-VL-7B + LoRA r=16, 2 epochs, LR=2e-5, bf16, flash_attn_2. H100 하나만. Qwen3.5-9B은 0.67 나옴.
**Ambiguity:** ~26.5%

### Round 5
**Q:** 0.87을 낼 때의 훈련 세부사항?
**A:** Full training code provided — train.csv only, 384px, batch_size=1, grad_accum=8, 2 epochs, label masking applied.
**Ambiguity:** ~20%

### Round 6 (Contrarian Mode)
**Q:** extract_choice() 파싱 실패율과 에러 패턴을 분석해본 적 있나요?
**A:** 분석 안 해봄
**Ambiguity:** ~15%

### Round 7
**Q:** SYSTEM_INSTRUCT와 build_mc_prompt() 내용이 뭐였나요?
**A:** 모르겠음 → baseline에서 확인: generic English VQA prompt
**Ambiguity:** ~12%

### Round 8 (Simplifier Mode)
**Q:** 혼자 작업하나요, 팀원 도움 가능?
**A:** 혼자 작업
**Ambiguity:** 9.5%
</details>
