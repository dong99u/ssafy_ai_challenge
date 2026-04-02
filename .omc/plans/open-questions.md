# Open Questions

## vqa-competition-top1 - 2026-04-02 (Delta Plan v5)

### Critical (Block Delta 1)
- [ ] Qwen3-VL-8B 4-bit actual peak VRAM on RTX 5060 Ti -- ~9B params is tighter than 7B. Hour 0 smoke test determines resolution or fallback to 7B.
- [ ] Qwen3-VL-8B model class in transformers -- May be Qwen3VLForConditionalGeneration or require AutoModelForVision2Seq. Check HuggingFace model card and transformers version.
- [ ] Does Qwen3-VL processor output mm_token_type_ids? -- If yes, the existing collator must pass it through to model.forward(). If collator drops unknown keys, this will silently break.
- [ ] Qwen3-VL-8B chat template format -- May differ from Qwen2.5-VL ChatML format. Label masking prefix boundary must still work correctly.

### Important (Block Delta 2)
- [ ] KVQA dataset format and usable subset size -- Need to inspect after download. How many of the 100K pairs are relevant to recycling/counting? 9,300 number-type questions claimed but needs verification.
- [ ] TOD dataset annotation format -- Need to inspect after download. What label schema is used? How many images per class? Can synthetic VQA generation match existing TrashNet/TACO quality?
- [ ] Optimal sample_weight for KVQA and TOD -- Current trashnet=0.55, taco_shared=0.45, taco_count=0.75. New data weights need tuning or A/B testing.

### Nice to Know (Do Not Block)
- [ ] Is bitsandbytes compatible with CUDA 13.0 / RTX 5060 Ti (Blackwell arch)? -- The baseline notebook ran on CUDA 12.8; CUDA 13.0 may have BNB compatibility issues that need testing. Already in existing notebook so likely fine.
- [ ] Qwen3-VL-8B Korean VQA performance -- No KOFFVQA benchmark exists. Real performance on Korean recycling VQA is unknown until Delta 1 validation gate.
- [ ] What is the current leaderboard top score? -- Calibrates whether gains from deltas are competitive.
- [ ] Does the competition limit number of submissions per day? -- Affects how aggressively we can iterate with leaderboard feedback.
- [ ] Does user have a Korean AI Hub account? -- AI Hub Recyclable Items (1M images) and Waste Plastic (800K images) require Korean national account. 1-3 day approval wait. Low priority vs instant-access KVQA+TOD.
- [ ] Whether cosine scheduler gives meaningful improvement over linear -- The notebook uses linear warmup. Cosine is a common improvement but may not matter with so few epochs.

### Resolved
- [x] ~~Label masking implementation~~ -- Already implemented in notebook (prefix masking with labels[i, :int(p_len)] = -100)
- [x] ~~Should choice-order augmentation be applied?~~ -- Already implemented (shuffle_options=True in training)
- [x] ~~Log-prob extraction token IDs~~ -- Already implemented in notebook with permutation-based log-prob ensemble
- [x] ~~Full training data usage~~ -- Already using all 5,073 samples
- [x] ~~Dev pseudo-label threshold~~ -- Already using 4/5 for shared, 3/5 for counting
- [x] ~~Korean system prompt~~ -- Already implemented (dual prompts: SYSTEM_PROMPT_GENERAL + SYSTEM_PROMPT_COUNT)
- [x] ~~OCR integration~~ -- Already implemented (EasyOCR + rapidfuzz + OCR ensemble channel)
- [x] ~~TrashNet/TACO synthetic data~~ -- Already implemented (max 2500 + 1800 samples)
- [x] ~~Qwen3-VL-8B VRAM overflow on 16GB~~ -- ELIMINATED. H100 80GB에서 BF16 8B = ~52GB. VRAM smoke test 불필요.
- [x] ~~bitsandbytes CUDA compatibility~~ -- BF16 training으로 전환. 4-bit quantization 불필요.

---

## vqa-competition-top1-h100 - 2026-04-02 (Delta Plan v6, H100 RALPLAN-DR)

### Critical (Block Delta 1)
- [ ] BF16 batch=4 peak VRAM on H100 -- Expected ~35-45GB but must verify. Determines batch size 4 vs 8.
- [ ] Colab H100 session 안정성 -- Disconnect 시 checkpoint resume 가능해야 함. Google Drive auto-save 설정 필요.

### Important (Block Delta 3)
- [ ] Full-text logprob scoring 구현 시 padding 처리 -- choice text 길이가 다르면 padding token의 log_prob이 score에 포함되지 않도록 masking 필요.
- [ ] Full-text logprob에서 choice text tokenization 일관성 -- "유리" vs "유리병" 등 choice text가 다른 token 수를 가짐. 정규화(평균 log_prob) vs 합산 결정 필요.

### Important (Block Delta 4)
- [ ] Qwen3-VL-8B model class in transformers -- Qwen3VLForConditionalGeneration 또는 AutoModelForVision2Seq. HuggingFace model card 및 transformers version 확인 필요.
- [ ] Qwen3-VL-8B chat template format -- Label masking prefix boundary가 정상 작동하는지 검증 필요.
- [ ] mm_token_type_ids collator 통과 여부 -- Qwen3-VL processor가 이 tensor를 생성하면 collator에서 명시적으로 포함시켜야 함.
- [ ] Qwen3-VL-8B Korean VQA 성능 -- Benchmark 없음. 1 epoch 후 validation accuracy로 판단.

### Nice to Know (Do Not Block)
- [ ] Optimal batch size for H100 (4 vs 8) -- Both work, 8이 더 빠르지만 effective batch size가 다름. A/B로 확인.
- [ ] Colab Pro/Pro+ GPU quota 잔여량 -- 4일간 H100 사용 가능 시간 확인.
- [ ] 제출 횟수 일일 제한 -- Leaderboard feedback 활용 전략에 영향.
- [ ] Cosine vs linear scheduler 실질 차이 -- Few-epoch training에서는 차이 미미할 수 있음.
- [ ] Ensemble weight optimization에서 validation set 크기 충분한지 -- Per-qtype optimization 시 sample 수가 작으면 overfitting.
