# Open Questions

## vqa-competition-notebook - 2026-04-04

- [ ] DATA_DIR path must be set to the actual H100 environment path -- executor must determine where data is mounted
- [ ] flash-attn installation may require specific CUDA toolkit version on H100 -- verify with `pip install flash-attn` before proceeding
- [ ] Token ID verification (Cell 12): if any of "a","b","c","d" maps to multiple tokens in the Qwen tokenizer, the logprob strategy needs adjustment (use first token only, or sum logprobs)
- [ ] num_workers=2 in DataLoader: if the H100 environment has limited CPU cores, this may need to be reduced to 0
- [ ] Ensemble tiebreaker currently defaults to Model A -- if Model B has lower val loss, the tiebreaker should use Model B instead (executor should check and adjust)
- [ ] If 672px causes OOM with LoRA r=64 (Model B), fallback to r=32 or reduce IMAGE_SIZE to 560px for Model B only
