# Open Questions

## vqa-competition-top1 - 2026-04-02 (Revised v2)

### Carried from v1 (still open)
- [ ] What is the current leaderboard top score? -- Calibrates whether 65-75% is competitive or if the task ceiling is lower/higher
- [ ] Does the competition limit number of submissions per day? -- Affects how aggressively we can iterate with leaderboard feedback
- [ ] Qwen2.5-VL-7B 4-bit actual peak VRAM on this specific hardware (RTX 5060 Ti) -- Need to verify on first training step; published estimates vary by driver/CUDA version
- [ ] Is bitsandbytes compatible with CUDA 13.1 / RTX 5060 Ti (Blackwell arch)? -- The baseline notebook ran on CUDA 12.8; CUDA 13.1 may have BNB compatibility issues that need testing

### New from v2 (Architect/Critic feedback)
- [ ] Exact token position of assistant answer in Qwen2.5-VL chat template -- Label masking requires knowing where the answer token is in the tokenized sequence. Must inspect actual tokenizer output to find the <|im_start|>assistant boundary.
- [ ] Actual counting/material split within the 902 high-confidence dev samples -- The 69.5/30.5 split is for ALL dev data. The >=4/5 subset may have a different distribution. Must verify before stratified subsampling.
- [ ] Log-prob extraction token IDs for "a","b","c","d" in Qwen2.5-VL tokenizer -- These may differ based on preceding whitespace/context. Need to test with actual tokenizer to get correct IDs.
- [ ] Whether 3B@672 is actually feasible in 16GB VRAM -- Planned as ensemble variant but not benchmarked. May need to fall back to 3B@504.

### Resolved from v1
- [x] ~~Should choice-order augmentation be applied at training time, inference time, or both?~~ -- Moved to Stretch Goal A, not core path
- [x] ~~For the dev pseudo-labels with agreement=3, should the minority answer be used as negative example?~~ -- Resolved: raised threshold to >=4/5 only, agreement=3 excluded entirely
- [x] ~~What is the actual inference speed per sample for 7B vs 3B?~~ -- Resolved: ensemble reduced to 2 models, inference time is feasible
