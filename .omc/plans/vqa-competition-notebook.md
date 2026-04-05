# VQA Competition Notebook Plan: 0.87 -> 0.93+ Accuracy

**Status**: REVISED (Architect feedback applied)
**Target**: Single Jupyter notebook (.ipynb) on H100 80GB
**Date**: 2026-04-04

---

## RALPLAN-DR Summary

### Principles (5)
1. **Data quality over quantity**: Filtered dev data with >=3/5 agreement is more valuable than noisy full dev set
2. **Resolution preserves signal**: 672px retains 3.1x more detail than 384px; counting questions (51.6%) require pixel-level accuracy
3. **Logprob inference eliminates parse errors**: Forward-pass logprob extraction is deterministic; no regex fallback needed
4. **Domain-specific prompting**: Korean recycling expert prompt outperforms generic English VQA prompt for Korean domain task
5. **TTA diversity over multi-model cost**: Answer-choice shuffling provides real prediction diversity at inference time without 7h training cost for a second model

### Decision Drivers (Top 3)
1. **Counting accuracy is the #1 lever** (51.6% of questions) -- higher resolution + counting-specific prompt instructions
2. **Dev data adds training rows** but only if filtered correctly (stratified to avoid distribution shift); corrected estimate: ~1,347 usable dev rows due to stratification bottleneck on underrepresented object_id and material types
3. **Logprob inference is strictly superior** to generate+parse (no default-to-"a" failures, faster inference)

### Options Considered

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A: Single model + TTA (CHOSEN)** | One Qwen2.5-VL-7B with all improvements + 4-permutation answer-choice TTA | Saves 7h training; real diversity via position debiasing; logprob averaging across permutations | 4x inference time (~2-3h); no model-level diversity |
| **B: Two-model ensemble** | Two models with different seeds/LR/LoRA rank, logprob averaging | Catches complementary errors, +1-2% from ensemble | ~7h extra training cost; tight time budget; marginal if TTA already debiases |
| **C: Larger model (Qwen2.5-VL-72B)** | Use 72B model with aggressive quantization | Higher base capability | May not fit H100 even with 4-bit; training extremely slow; competition rules may limit |

**Why A**: TTA with answer-choice shuffling eliminates position bias (a known failure mode in MC-VQA) without the 7h cost of training a second model. Logprob averaging across 4 permutations provides genuine prediction diversity. If time remains after Model A + TTA, Model B can still be trained as a bonus.

**Option B deferral**: Not invalidated -- simply deprioritized. If >6h remain after Phase 4 (TTA inference), train Model B and logprob-average the ensemble. This is now in the "Optional" section.

**Option C invalidation**: Qwen2.5-VL-72B in 4-bit NF4 requires ~38GB for weights alone. With 672px images, KV cache, and gradient checkpointing activations, peak memory would exceed 80GB during training. Even if it fits, training would take 16-20h for a single model, leaving no room for error recovery.

---

## ADR (Architecture Decision Record)

- **Decision**: Single Qwen2.5-VL-7B-Instruct model with logprob inference, 672px resolution, domain prompt, filtered dev data, and TTA via answer-choice shuffling (4 permutations with logprob averaging)
- **Drivers**: Counting accuracy leverage, dev data quality, parse-error elimination, position-bias elimination via TTA
- **Alternatives considered**: Two-model ensemble (deferred to optional phase if time permits), 72B model (memory/time risk -- invalidated)
- **Why chosen**: Best accuracy-to-time ratio. TTA provides diversity without doubling training time. Single model + TTA fits comfortably in ~10-13h, leaving margin for optional Model B or reruns.
- **Consequences**: ~5-8h training time for one model, ~2-3h TTA inference, single model checkpoint saved; if Model B is trained, ensemble logic adds ~20 lines
- **Follow-ups**: If TTA accuracy < 0.93, train Model B (optional phase). If ensemble < 0.93, consider (1) increasing resolution to 784px, (2) training a third model with different LoRA targets

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| OOM at 672px with r=32 LoRA | Medium | High | Gradient checkpointing + grad_accum=16; GPU memory probe after step 1; fallback to 560px if >70GB |
| Dev data hurts due to noise | Low | Medium | Validation monitors per-source accuracy; can ablate dev rows |
| Logprob tokens not in vocabulary as expected | Low | High | Pre-verify token IDs for "a","b","c","d" before training |
| Training diverges with cosine schedule | Low | Medium | Warmup ratio 0.05, gradient clipping 1.0, monitor val loss each epoch |
| Training too slow for H100 | Medium | Medium | Timing checkpoint after step 100; if per-step >2.5s, reduce to 2 epochs or 560px |
| Val accuracy too low (<0.87) | Low | High | Fallback guard: do NOT use model predictions if val accuracy < 0.87 |

---

## TTA Strategy: Answer-Choice Shuffling

For each test question, run 4 forward passes with different answer orderings:

| Permutation | Ordering |
|-------------|----------|
| Original | (a)A (b)B (c)C (d)D |
| Shuffled 1 | (a)B (b)D (c)A (d)C |
| Shuffled 2 | (a)C (b)A (c)D (d)B |
| Shuffled 3 | (a)D (b)C (c)B (d)A |

For each pass:
1. Extract logprobs for tokens a, b, c, d
2. Map each logprob back to the **original answer option** it corresponds to
3. Average logprobs across all 4 permutations per original answer option
4. Pick the original answer option with the highest averaged logprob

This eliminates position bias in the model's predictions.

---

## Time Estimates

| Phase | Time | Notes |
|-------|------|-------|
| Phase 1: Data prep + notebook setup | 1h | Cells 1-9 |
| Phase 2: Model A training (3 epochs, ~6,000 rows) | 5-8h | Timing checkpoint after 100 steps; early stopping patience=1 |
| Phase 3: Logprob inference + error analysis | 45min-1h | Forward pass on val set + test set |
| Phase 4: TTA inference (5,074 x 4 permutations) | 2-3h | 4x regular inference time |
| Phase 5 (optional): Model B training | 5-8h | Only if >6h remain after Phase 4 |
| Phase 6: Generate submissions | 30min | Single model, TTA, optionally ensemble |
| **Total (without Model B)** | **~10-13h** | Comfortable within 24h budget |
| **Total (with Model B)** | **~18-22h** | Tight but feasible |

---

## Detailed Implementation Plan

### Cell 1: Environment Setup

**What**: Install all required packages with pinned versions.

**Cell content requirements**:
- `!pip install` with: `transformers>=4.49.0`, `accelerate>=0.34.2`, `peft>=0.13.2`, `bitsandbytes>=0.45.0`, `torch>=2.5.0`, `pillow`, `pandas`, `scikit-learn`, `tqdm`, `flash-attn` (for flash_attention_2)
- `import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))`
- Verify flash-attn: `import flash_attn; print(flash_attn.__version__)`

**Acceptance criteria**:
- [ ] Cell runs without error
- [ ] torch.cuda.is_available() returns True
- [ ] flash_attn imports successfully

---

### Cell 2: Imports and Constants

**What**: All imports and hyperparameter constants in one place for easy tuning.

**Cell content requirements**:
```python
import os, re, math, random, json, gc, time, itertools
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import torch
from typing import Dict, List, Any, Optional, Tuple
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm.auto import tqdm

Image.MAX_IMAGE_PIXELS = None
device = "cuda"

# -- Hyperparameters (Model A) --
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_SIZE = 672
SEED_A = 42
BATCH_SIZE = 1
GRAD_ACCUM = 8
NUM_EPOCHS = 3
LR_A = 2e-5
LORA_R_A = 32
LORA_ALPHA_A = 64
LORA_DROPOUT = 0.05
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0
DEV_AGREEMENT_THRESHOLD = 3  # out of 5 annotators
EARLY_STOPPING_PATIENCE = 1  # stop if val loss increases for 1 consecutive epoch
TIMING_CHECKPOINT_STEP = 100  # print time estimate after this many steps
GPU_MEMORY_WARN_THRESHOLD_GB = 70  # warn if peak memory > this after step 1
VAL_ACCURACY_FALLBACK_THRESHOLD = 0.87  # do NOT use model if val acc < this

# -- Optional Model B Hyperparameters (Phase 5) --
SEED_B = 123
LR_B = 5e-5
LORA_R_B = 64
LORA_ALPHA_B = 128

# -- TTA Permutations --
TTA_PERMUTATIONS = [
    ["a", "b", "c", "d"],  # original
    ["b", "d", "a", "c"],  # shuffled 1
    ["c", "a", "d", "b"],  # shuffled 2
    ["d", "c", "b", "a"],  # shuffled 3
]

# -- Paths (adjust DATA_DIR to your environment) --
DATA_DIR = "/path/to/data"
TRAIN_CSV = f"{DATA_DIR}/csv/train.csv"
DEV_CSV = f"{DATA_DIR}/csv/dev.csv"
TEST_CSV = f"{DATA_DIR}/csv/test.csv"
SAVE_DIR_A = "./checkpoints/model_a"
SAVE_DIR_B = "./checkpoints/model_b"
```

**Acceptance criteria**:
- [ ] All imports succeed
- [ ] Constants are clearly named and grouped
- [ ] EARLY_STOPPING_PATIENCE, TIMING_CHECKPOINT_STEP, GPU_MEMORY_WARN_THRESHOLD_GB, VAL_ACCURACY_FALLBACK_THRESHOLD are defined
- [ ] TTA_PERMUTATIONS has 4 permutations
- [ ] Executor adjusts DATA_DIR to actual data location

---

### Cell 3: Data Loading and Dev Filtering

**What**: Load train.csv and dev.csv. Apply majority vote to dev.csv, keep only rows with >=3/5 agreement.

**Cell content requirements**:

1. Load train_df and dev_df with pandas
2. Fix image paths to absolute: `df["path"] = df["path"].apply(lambda p: os.path.join(DATA_DIR, p))`
3. For dev_df, compute majority vote:
   ```python
   def majority_vote(row):
       votes = [row[f"answer{i}"] for i in range(1, 6)]
       counter = Counter(votes)
       most_common, count = counter.most_common(1)[0]
       return most_common if count >= DEV_AGREEMENT_THRESHOLD else None

   dev_df["answer"] = dev_df.apply(majority_vote, axis=1)
   dev_df = dev_df.dropna(subset=["answer"])
   ```
4. Keep only columns matching train_df: `["id", "path", "question", "a", "b", "c", "d", "answer"]`
5. Print statistics:
   - train_df shape, dev_df shape after filtering
   - Agreement distribution: how many had 3/5, 4/5, 5/5

**Acceptance criteria**:
- [ ] dev_df retains rows with >=3/5 agreement
- [ ] dev_df has same columns as train_df
- [ ] No NaN values in answer column
- [ ] Image paths are absolute and valid

---

### Cell 4: Question Type Classification

**What**: Classify each question into types (counting, material, object_id, other) for stratification.

**Cell content requirements**:
```python
def classify_question(question: str) -> str:
    q = question.lower()
    counting_keywords = ["몇 개", "몇개", "개수", "몇 병", "몇 캔", "총 몇"]
    material_keywords = ["재질", "재료", "소재", "무엇으로 만들", "어떤 재질"]

    if any(kw in q for kw in counting_keywords):
        return "counting"
    elif any(kw in q for kw in material_keywords):
        return "material"
    else:
        if "무엇" in q or "어떤 것" in q or "종류" in q:
            return "object_id"
        return "other"

train_df["q_type"] = train_df["question"].apply(classify_question)
dev_df["q_type"] = dev_df["question"].apply(classify_question)
```

Print distribution for both datasets. Key insight: train has ~34.5% counting, dev has ~71.3% counting. Must subsample dev to match train distribution.

**Acceptance criteria**:
- [ ] Every row has a q_type in {counting, material, object_id, other}
- [ ] Distribution printed for both train and dev

---

### Cell 5: Dev Data Stratified Sampling and Merge

**What**: Subsample dev data so its question-type distribution matches train, then merge.

**NOTE**: The stratification bottleneck is on underrepresented types (object_id, material) in dev. Expected usable dev rows: ~1,347 (not ~3,430 as previously estimated). Total combined dataset: ~6,000-6,500 rows.

**Cell content requirements**:

1. Compute train distribution ratios per q_type
2. For each q_type, sample from dev_df proportionally:
   ```python
   train_dist = train_df["q_type"].value_counts(normalize=True)

   # Find the bottleneck q_type (smallest dev_count / train_ratio)
   dev_type_counts = dev_df["q_type"].value_counts()
   max_total = min(
       dev_type_counts[qt] / train_dist[qt]
       for qt in train_dist.index
       if qt in dev_type_counts
   )
   max_total = int(max_total)

   sampled_dev = []
   for qt in train_dist.index:
       n = int(max_total * train_dist[qt])
       qt_df = dev_df[dev_df["q_type"] == qt]
       if len(qt_df) >= n:
           sampled_dev.append(qt_df.sample(n=n, random_state=SEED_A))
       else:
           sampled_dev.append(qt_df)

   dev_sampled = pd.concat(sampled_dev, ignore_index=True)
   ```
3. Merge: `combined_df = pd.concat([train_df, dev_sampled], ignore_index=True)`
4. Drop the q_type column after merge (it was only for stratification)
5. Print final combined_df shape and distribution
6. Print bottleneck q_type and how many dev rows were used

**Acceptance criteria**:
- [ ] combined_df has train rows + sampled dev rows (expected ~5,073 + ~1,000-1,500 = ~6,000-6,500 total)
- [ ] Question type distribution of combined_df is close to train_df distribution (within 5% per category)
- [ ] No duplicate rows between train and dev portions
- [ ] Bottleneck type identified and printed

---

### Cell 6: Train/Val Split

**What**: 90/10 stratified split using question type as stratification key.

**Cell content requirements**:
```python
combined_df["q_type"] = combined_df["question"].apply(classify_question)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED_A)
train_idx, val_idx = next(sss.split(combined_df, combined_df["q_type"]))

final_train_df = combined_df.iloc[train_idx].reset_index(drop=True)
final_val_df = combined_df.iloc[val_idx].reset_index(drop=True)

print(f"Train: {len(final_train_df)}, Val: {len(final_val_df)}")
print(f"Train q_type dist:\n{final_train_df['q_type'].value_counts(normalize=True)}")
print(f"Val q_type dist:\n{final_val_df['q_type'].value_counts(normalize=True)}")
```

**Acceptance criteria**:
- [ ] Val set is ~10% of combined data (~600-650 rows)
- [ ] Question type distribution matches between train and val (stratified)
- [ ] No data leakage (same image not in both train and val)

---

### Cell 7: Domain-Specific System Prompt

**What**: Korean-language system prompt specialized for recycling VQA.

**Cell content requirements**:
```python
SYSTEM_PROMPT = (
    "당신은 재활용 분류 전문가입니다. "
    "재활용품 이미지를 보고 질문에 정확히 답변합니다.\n\n"
    "핵심 지침:\n"
    "1. 개수를 세는 질문: 이미지를 꼼꼼히 살펴보고 해당 물품의 정확한 개수를 세세요. "
    "겹치거나 부분적으로 보이는 물품도 포함합니다.\n"
    "2. 재질 판별: 플라스틱(투명/불투명, 유연/경질), 유리(투명/색유리), "
    "금속(알루미늄/철), 종이(골판지/일반), 비닐/스티로폼을 구분하세요.\n"
    "3. 금속 vs 플라스틱: 광택, 반사, 찌그러짐 패턴으로 구분합니다. "
    "금속은 광택이 있고 찌그러지면 주름이 생깁니다. "
    "플라스틱은 무광이며 라벨이 붙어있는 경우가 많습니다.\n\n"
    "반드시 a, b, c, d 중 하나의 소문자 한 글자로만 답하세요."
)

def build_mc_prompt(question: str, a: str, b: str, c: str, d: str) -> str:
    return (
        f"{question}\n"
        f"(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n"
        "정답을 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
    )

def build_mc_prompt_shuffled(
    question: str, options: dict, permutation: list
) -> Tuple[str, dict]:
    """Build MC prompt with shuffled answer positions.

    Args:
        question: The question text
        options: Dict mapping original keys to option text, e.g. {"a": "텍스트1", ...}
        permutation: List of original keys in shuffled order, e.g. ["b", "d", "a", "c"]

    Returns:
        (prompt_text, position_to_original_map)
        position_to_original_map: {"a": "b", "b": "d", ...} meaning position "a" holds original option "b"
    """
    position_labels = ["a", "b", "c", "d"]
    position_to_original = {}
    lines = [question]
    for pos_label, orig_key in zip(position_labels, permutation):
        lines.append(f"({pos_label}) {options[orig_key]}")
        position_to_original[pos_label] = orig_key
    lines.append("")
    lines.append("정답을 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요.")
    return "\n".join(lines), position_to_original
```

**Acceptance criteria**:
- [ ] System prompt is in Korean
- [ ] Includes counting strategy, material categories, metal-vs-plastic disambiguation
- [ ] build_mc_prompt returns properly formatted string
- [ ] build_mc_prompt_shuffled returns (prompt, mapping) tuple
- [ ] Mapping correctly tracks which original answer sits in which position

---

### Cell 8: VQAMCDataset Class

**What**: Custom Dataset with proper structure for the collator.

**Cell content requirements**:
```python
class VQAMCDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")

        user_text = build_mc_prompt(
            str(row["question"]),
            str(row["a"]), str(row["b"]),
            str(row["c"]), str(row["d"]),
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_text},
            ]},
        ]
        if self.train:
            gold = str(row["answer"]).strip().lower()
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": gold}]}
            )

        return {"messages": messages, "image": img}
```

**Acceptance criteria**:
- [ ] Returns dict with "messages" and "image" keys
- [ ] In train mode, assistant message contains single letter answer
- [ ] Image is loaded as RGB PIL Image

---

### Cell 9: DataCollator with Label Masking

**What**: Collator that applies chat template, tokenizes, and masks labels so loss is only computed on the assistant answer token.

**Cell content requirements**:
```python
@dataclass
class VQACollator:
    processor: Any
    train: bool = True

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        texts, images = [], []
        for sample in batch:
            text = self.processor.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=(not self.train),
            )
            texts.append(text)
            images.append(sample["image"])

        enc = self.processor(
            text=texts, images=images,
            padding=True, return_tensors="pt",
        )

        if self.train:
            labels = enc["input_ids"].clone()
            # Mask pad tokens
            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100

            # For each sequence, mask all prompt tokens (keep only assistant answer)
            for i in range(labels.shape[0]):
                prompt_msgs = batch[i]["messages"][:-1]  # without assistant
                prompt_text = self.processor.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True,
                )
                prompt_enc = self.processor.tokenizer(
                    prompt_text, return_tensors="pt", add_special_tokens=False,
                )
                prompt_len = prompt_enc["input_ids"].shape[1]
                labels[i, :prompt_len] = -100

            enc["labels"] = labels

        return enc
```

**Key design decision**: Label masking ensures the model only learns to predict the answer letter, not to regenerate the entire prompt. This focuses the gradient signal and typically improves convergence.

**Acceptance criteria**:
- [ ] Labels have -100 for all non-answer positions
- [ ] Only the answer token(s) + EOS have real label values
- [ ] Pad tokens are masked with -100

---

### Cell 10: Model Loading Function

**What**: Reusable function to load model with given LoRA config. Used for both Model A and (optionally) Model B.

**Cell content requirements**:
```python
def load_model_and_processor(
    model_id: str, image_size: int,
    lora_r: int, lora_alpha: int,
    lora_dropout: float, seed: int,
):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=image_size * image_size,
        max_pixels=image_size * image_size,
        trust_remote_code=True,
    )

    base_model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def load_model_for_inference(save_dir: str):
    """Load base model + saved LoRA adapter for inference."""
    processor = AutoProcessor.from_pretrained(
        save_dir,
        min_pixels=IMAGE_SIZE * IMAGE_SIZE,
        max_pixels=IMAGE_SIZE * IMAGE_SIZE,
        trust_remote_code=True,
    )
    base_model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, save_dir)
    model.eval()
    return model, processor
```

**Important**: No quantization (bf16 full weights). H100 80GB can hold Qwen2.5-VL-7B (~15GB) + LoRA adapters + activations at 672px comfortably in bf16. Removing quantization avoids the quality loss from NF4.

**Acceptance criteria**:
- [ ] Model loads without OOM
- [ ] flash_attention_2 is active
- [ ] Trainable parameters printed (~50-100M depending on r)
- [ ] gradient_checkpointing is enabled
- [ ] load_model_for_inference correctly loads saved adapter

---

### Cell 11: Training Function (with Early Stopping, Timing Checkpoint, Memory Probe)

**What**: Complete training function with cosine schedule, gradient clipping, best-model saving, per-epoch validation, early stopping (patience=1), timing checkpoint after step 100, and GPU memory probe after step 1.

**Cell content requirements**:
```python
def train_model(
    model, processor,
    train_df: pd.DataFrame, val_df: pd.DataFrame,
    num_epochs: int, lr: float, grad_accum: int,
    save_dir: str, seed: int,
):
    random.seed(seed)
    torch.manual_seed(seed)

    train_ds = VQAMCDataset(train_df, processor, train=True)
    val_ds = VQAMCDataset(val_df, processor, train=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=VQACollator(processor, train=True),
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=VQACollator(processor, train=True),
        num_workers=2, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = num_epochs * math.ceil(len(train_loader) / grad_accum)
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(save_dir, exist_ok=True)
    train_start_time = time.time()

    for epoch in range(num_epochs):
        # -- Train --
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]")

        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum

            loss.backward()
            running_loss += loss.item()

            # === GPU Memory Probe after step 1 ===
            if epoch == 0 and step == 1:
                peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
                print(f"\n[MEMORY PROBE] Peak GPU memory after step 1: {peak_mem_gb:.1f}GB")
                if peak_mem_gb > GPU_MEMORY_WARN_THRESHOLD_GB:
                    print(f"  WARNING: Peak memory ({peak_mem_gb:.1f}GB) exceeds {GPU_MEMORY_WARN_THRESHOLD_GB}GB threshold!")
                    print(f"  Consider fallback: reduce IMAGE_SIZE to 560px or reduce LORA_R")
                else:
                    print(f"  OK: Memory usage within safe range ({peak_mem_gb:.1f}GB / 80GB)")

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), MAX_GRAD_NORM
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix({
                    "loss": f"{running_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })
                running_loss = 0.0

            # === Timing Checkpoint after step 100 ===
            if epoch == 0 and step == TIMING_CHECKPOINT_STEP:
                elapsed = time.time() - train_start_time
                per_step = elapsed / step
                total_train_steps = num_epochs * len(train_loader)
                estimated_total = per_step * total_train_steps
                print(f"\n[TIMING CHECKPOINT] After {step} steps:")
                print(f"  Per-step time: {per_step:.2f}s")
                print(f"  Estimated total training time: {estimated_total/3600:.1f}h")
                print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
                if per_step > 2.5:
                    print(f"  WARNING: Per-step time ({per_step:.2f}s) exceeds 2.5s threshold!")
                    print(f"  Consider: reduce NUM_EPOCHS to 2, or reduce IMAGE_SIZE to 560px")

        # Handle remaining gradient accumulation steps
        if step % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), MAX_GRAD_NORM
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # -- Validate --
        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for vb in tqdm(val_loader, desc=f"Epoch {epoch+1} [valid]"):
                    vb = {k: v.to(device) for k, v in vb.items()}
                    val_loss += model(**vb).loss.item()
                    val_steps += 1

        avg_val_loss = val_loss / max(val_steps, 1)
        print(f"[Epoch {epoch+1}] val_loss={avg_val_loss:.4f} | best={best_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            print(f"  -> Saved best model to {save_dir}")
        else:
            patience_counter += 1
            print(f"  -> Val loss did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  -> EARLY STOPPING triggered at epoch {epoch+1}")
                break

    total_train_time = time.time() - train_start_time
    print(f"\nTraining completed in {total_train_time/3600:.1f}h")
    return best_val_loss
```

**Acceptance criteria**:
- [ ] Training loss decreases over epochs
- [ ] Validation loss is computed each epoch
- [ ] Best model is saved based on lowest val loss
- [ ] Gradient clipping is applied (MAX_GRAD_NORM=1.0)
- [ ] Cosine schedule with warmup is active
- [ ] GPU memory probe prints after step 1; warns if >70GB
- [ ] Timing checkpoint prints after step 100; warns if per-step >2.5s
- [ ] Early stopping triggers if val loss increases for EARLY_STOPPING_PATIENCE consecutive epochs

---

### Cell 12: Token ID Verification

**What**: Pre-verify that "a", "b", "c", "d" each map to a single token ID. Critical for logprob inference.

**Cell content requirements**:
```python
temp_processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = temp_processor.tokenizer

for letter in ["a", "b", "c", "d"]:
    ids = tokenizer.encode(letter, add_special_tokens=False)
    print(f"'{letter}' -> token_ids={ids}, decoded='{tokenizer.decode(ids)}'")
    assert len(ids) == 1, f"Expected single token for '{letter}', got {ids}"

ANSWER_TOKEN_IDS = {
    letter: tokenizer.encode(letter, add_special_tokens=False)[0]
    for letter in ["a", "b", "c", "d"]
}
print(f"\nAnswer token ID mapping: {ANSWER_TOKEN_IDS}")
del temp_processor, tokenizer
gc.collect()
```

**Acceptance criteria**:
- [ ] Each of "a","b","c","d" maps to exactly one token
- [ ] ANSWER_TOKEN_IDS dict has 4 entries
- [ ] Token IDs are valid integers

---

### Cell 13: Train Model A

**What**: Load and train the first model (seed=42, lr=2e-5, LoRA r=32).

**Cell content requirements**:
```python
print("=" * 60)
print("TRAINING MODEL A: seed=42, lr=2e-5, LoRA r=32, alpha=64")
print(f"Dataset: {len(final_train_df)} train, {len(final_val_df)} val")
print(f"Epochs: {NUM_EPOCHS} (early stopping patience={EARLY_STOPPING_PATIENCE})")
print("=" * 60)

model_a, processor_a = load_model_and_processor(
    MODEL_ID, IMAGE_SIZE,
    lora_r=LORA_R_A, lora_alpha=LORA_ALPHA_A,
    lora_dropout=LORA_DROPOUT, seed=SEED_A,
)

best_loss_a = train_model(
    model_a, processor_a,
    final_train_df, final_val_df,
    num_epochs=NUM_EPOCHS, lr=LR_A,
    grad_accum=GRAD_ACCUM, save_dir=SAVE_DIR_A,
    seed=SEED_A,
)
print(f"Model A best val loss: {best_loss_a:.4f}")
```

**Acceptance criteria**:
- [ ] Training completes without OOM
- [ ] Best model saved to SAVE_DIR_A
- [ ] Val loss is reported
- [ ] Memory probe and timing checkpoint executed during training

---

### Cell 14: Logprob Inference Function

**What**: Define the logprob inference function used for both validation and test. Returns both predictions and raw logprobs (needed for TTA averaging).

**Cell content requirements**:
```python
def logprob_inference(
    model, processor,
    df: pd.DataFrame,
    answer_token_ids: dict,
    return_logprobs: bool = False,
) -> List[str] | Tuple[List[str], List[dict]]:
    """Forward-pass logprob extraction instead of generate().

    If return_logprobs=True, returns (predictions, logprobs_list)
    where logprobs_list[i] = {"a": float, "b": float, "c": float, "d": float}
    """
    model.eval()
    predictions = []
    all_logprobs = []
    ds = VQAMCDataset(df, processor, train=False)

    for idx in tqdm(range(len(ds)), desc="Logprob Inference"):
        sample = ds[idx]
        text = processor.apply_chat_template(
            sample["messages"], tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text], images=[sample["image"]],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)

        # Logits at last position
        last_logits = outputs.logits[0, -1, :]

        # Extract logprobs for a, b, c, d
        logprobs = {
            letter: last_logits[tid].item()
            for letter, tid in answer_token_ids.items()
        }

        pred = max(logprobs, key=logprobs.get)
        predictions.append(pred)
        if return_logprobs:
            all_logprobs.append(logprobs)

    if return_logprobs:
        return predictions, all_logprobs
    return predictions
```

**Acceptance criteria**:
- [ ] Returns list of strings, each in {a, b, c, d}
- [ ] Uses forward pass (not generate)
- [ ] When return_logprobs=True, also returns per-sample logprob dicts
- [ ] Handles all rows without error

---

### Cell 15: Validation Accuracy + Error Analysis (Model A) + Fallback Guard

**What**: Compute accuracy on val set using logprob inference. Break down by question type. Apply fallback guard if accuracy < 0.87.

**Cell content requirements**:
```python
# Reload best Model A
del model_a; gc.collect(); torch.cuda.empty_cache()
model_a_best, proc_a = load_model_for_inference(SAVE_DIR_A)

val_preds_a = logprob_inference(model_a_best, proc_a, final_val_df, ANSWER_TOKEN_IDS)

# Overall accuracy
val_gold = final_val_df["answer"].str.strip().str.lower().tolist()
correct_a = sum(p == g for p, g in zip(val_preds_a, val_gold))
val_accuracy_a = correct_a / len(val_gold)
print(f"\nModel A Val Accuracy: {correct_a}/{len(val_gold)} = {val_accuracy_a:.4f}")

# === FALLBACK GUARD ===
if val_accuracy_a < VAL_ACCURACY_FALLBACK_THRESHOLD:
    print(f"\n{'='*60}")
    print(f"FALLBACK GUARD TRIGGERED: Val accuracy ({val_accuracy_a:.4f}) < {VAL_ACCURACY_FALLBACK_THRESHOLD}")
    print(f"WARNING: Do NOT use this model's predictions for final submission!")
    print(f"Consider: re-check data pipeline, reduce LR, increase epochs")
    print(f"{'='*60}")
else:
    print(f"\nFallback guard OK: Val accuracy ({val_accuracy_a:.4f}) >= {VAL_ACCURACY_FALLBACK_THRESHOLD}")

# Per question-type breakdown
final_val_df["pred_a"] = val_preds_a
final_val_df["correct_a"] = (
    final_val_df["pred_a"] == final_val_df["answer"].str.strip().str.lower()
)
print("\nPer-type accuracy:")
print(final_val_df.groupby("q_type")["correct_a"].mean())

# Error confusion: predicted vs actual answer position
confusion = Counter()
for _, row in final_val_df.iterrows():
    actual = row["answer"].strip().lower()
    pred = row["pred_a"]
    confusion[(actual, pred)] += 1

print("\nErrors (actual -> predicted): count")
for (actual, pred), cnt in sorted(confusion.items()):
    if actual != pred:
        print(f"  {actual} -> {pred}: {cnt}")
```

**Acceptance criteria**:
- [ ] Val accuracy printed (target: > 0.90 for Model A alone)
- [ ] Per question-type accuracy is shown for counting, material, object_id, other
- [ ] Error confusion matrix identifies systematic mispredictions
- [ ] Fallback guard prints warning if val accuracy < 0.87

---

### Cell 16: TTA Inference Function

**What**: Define the TTA inference function that runs 4 forward passes per question with shuffled answer positions, then averages logprobs mapped back to original answer options.

**Cell content requirements**:
```python
def tta_logprob_inference(
    model, processor,
    df: pd.DataFrame,
    answer_token_ids: dict,
    permutations: list,
) -> Tuple[List[str], List[dict]]:
    """TTA with answer-choice shuffling.

    For each question, runs len(permutations) forward passes with different
    answer orderings. Logprobs are mapped back to original answer options
    and averaged across permutations.

    Returns:
        (predictions, averaged_logprobs)
    """
    model.eval()
    predictions = []
    averaged_logprobs_list = []

    for idx in tqdm(range(len(df)), desc="TTA Inference"):
        row = df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        options = {
            "a": str(row["a"]),
            "b": str(row["b"]),
            "c": str(row["c"]),
            "d": str(row["d"]),
        }

        # Accumulate logprobs per original answer option
        orig_logprobs = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}

        for perm in permutations:
            prompt_text, pos_to_orig = build_mc_prompt_shuffled(
                str(row["question"]), options, perm
            )
            # orig_to_pos: reverse mapping
            orig_to_pos = {v: k for k, v in pos_to_orig.items()}

            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_text},
                ]},
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = processor(
                text=[text], images=[img],
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(**inputs)

            last_logits = outputs.logits[0, -1, :]

            # Get logprobs for position labels a,b,c,d
            for pos_label in ["a", "b", "c", "d"]:
                logprob = last_logits[answer_token_ids[pos_label]].item()
                # Map position back to original answer option
                orig_key = pos_to_orig[pos_label]
                orig_logprobs[orig_key] += logprob

        # Average across permutations
        n_perm = len(permutations)
        avg_logprobs = {k: v / n_perm for k, v in orig_logprobs.items()}

        pred = max(avg_logprobs, key=avg_logprobs.get)
        predictions.append(pred)
        averaged_logprobs_list.append(avg_logprobs)

    return predictions, averaged_logprobs_list
```

**Acceptance criteria**:
- [ ] Runs 4 forward passes per question (one per permutation)
- [ ] Logprobs are correctly mapped back to original answer options
- [ ] Averaging is over all permutations
- [ ] Final prediction is the original answer key with highest averaged logprob
- [ ] Returns both predictions and averaged logprob dicts

---

### Cell 17: Test Inference - Single Model (Model A)

**What**: Run logprob inference on test set for Model A (no TTA). This is the baseline submission.

**Cell content requirements**:
```python
test_df = pd.read_csv(TEST_CSV)
test_df["path"] = test_df["path"].apply(lambda p: os.path.join(DATA_DIR, p))

# Model A is already loaded from Cell 15
test_preds_a = logprob_inference(model_a_best, proc_a, test_df, ANSWER_TOKEN_IDS)
print(f"Model A test predictions: {Counter(test_preds_a)}")

sub_a = pd.DataFrame({"id": test_df["id"], "answer": test_preds_a})
sub_a.to_csv("submission_model_a.csv", index=False)
print("Saved submission_model_a.csv")
```

**Acceptance criteria**:
- [ ] 5,074 predictions generated
- [ ] All predictions in {a, b, c, d}
- [ ] CSV saved with correct format (id, answer)

---

### Cell 18: TTA Inference on Test Set

**What**: Run TTA with 4 answer-choice permutations on the test set. This is the primary submission.

**Cell content requirements**:
```python
print("=" * 60)
print(f"TTA INFERENCE: {len(test_df)} rows x {len(TTA_PERMUTATIONS)} permutations")
print(f"Estimated time: {len(test_df) * len(TTA_PERMUTATIONS) * 0.5 / 60:.0f}-{len(test_df) * len(TTA_PERMUTATIONS) * 1.5 / 60:.0f} min")
print("=" * 60)

tta_start = time.time()
test_preds_tta, test_logprobs_tta = tta_logprob_inference(
    model_a_best, proc_a, test_df, ANSWER_TOKEN_IDS, TTA_PERMUTATIONS,
)
tta_elapsed = time.time() - tta_start
print(f"\nTTA inference completed in {tta_elapsed/3600:.1f}h ({tta_elapsed/60:.0f} min)")
print(f"TTA predictions: {Counter(test_preds_tta)}")

# Compare with single-model predictions
agree = sum(a == t for a, t in zip(test_preds_a, test_preds_tta))
print(f"\nSingle vs TTA agreement: {agree}/{len(test_preds_a)} = {agree/len(test_preds_a):.4f}")
print(f"TTA changed {len(test_preds_a) - agree} predictions")

sub_tta = pd.DataFrame({"id": test_df["id"], "answer": test_preds_tta})
sub_tta.to_csv("submission_tta.csv", index=False)
print("Saved submission_tta.csv")
```

**Acceptance criteria**:
- [ ] 5,074 predictions generated
- [ ] All predictions in {a, b, c, d}
- [ ] Agreement rate with single-model is printed (expect >90%)
- [ ] TTA inference completes within 3h
- [ ] submission_tta.csv saved

---

### Cell 19: Memory Cleanup

**What**: Free GPU memory.

```python
del model_a_best, proc_a
gc.collect()
torch.cuda.empty_cache()
print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1e9:.1f}GB")
```

**Acceptance criteria**:
- [ ] GPU memory drops below 2GB

---

### Cell 20: Final Submission + Sanity Checks

**What**: Generate final submission CSV and run sanity checks.

```python
# Use TTA submission as the primary submission
import shutil
shutil.copy("submission_tta.csv", "submission.csv")
print("Final submission: submission.csv (TTA-based)")

# Sanity checks
submission = pd.read_csv("submission.csv")
print(f"\nSubmission shape: {submission.shape}")
print(f"Prediction distribution: {Counter(submission['answer'].tolist())}")
print(submission.head(10))

assert len(submission) == len(test_df), f"Row count mismatch: {len(submission)} vs {len(test_df)}"
assert set(submission["answer"].unique()).issubset({"a","b","c","d"}), "Invalid answers!"
assert submission["id"].nunique() == len(submission), "Duplicate IDs!"
print("\nAll sanity checks passed.")

# Summary of all generated submissions
print("\n" + "=" * 60)
print("GENERATED SUBMISSIONS:")
print(f"  1. submission_model_a.csv  - Single model (Model A), no TTA")
print(f"  2. submission_tta.csv      - Model A + TTA (4 permutations)")
print(f"  3. submission.csv          - Final (copy of submission_tta.csv)")
print("=" * 60)
```

**Acceptance criteria**:
- [ ] submission.csv has exactly 5,074 rows
- [ ] Two columns: id, answer
- [ ] All answers in {a, b, c, d}
- [ ] No duplicate IDs
- [ ] Three CSV files generated: submission_model_a.csv, submission_tta.csv, submission.csv

---

## OPTIONAL: Phase 5 - Model B + Ensemble (if >6h remain)

The following cells should ONLY be run if there is sufficient time remaining (>6 hours) after Phase 4 (TTA inference). Model B provides model-level diversity on top of TTA's position-debiasing.

### Optional Cell A: Train Model B

```python
print("=" * 60)
print("OPTIONAL: TRAINING MODEL B: seed=123, lr=5e-5, LoRA r=64, alpha=128")
print("=" * 60)

model_b, processor_b = load_model_and_processor(
    MODEL_ID, IMAGE_SIZE,
    lora_r=LORA_R_B, lora_alpha=LORA_ALPHA_B,
    lora_dropout=LORA_DROPOUT, seed=SEED_B,
)

best_loss_b = train_model(
    model_b, processor_b,
    final_train_df, final_val_df,
    num_epochs=NUM_EPOCHS, lr=LR_B,
    grad_accum=GRAD_ACCUM, save_dir=SAVE_DIR_B,
    seed=SEED_B,
)
print(f"Model B best val loss: {best_loss_b:.4f}")
```

### Optional Cell B: Model B Validation

```python
del model_b; gc.collect(); torch.cuda.empty_cache()
model_b_best, proc_b = load_model_for_inference(SAVE_DIR_B)

val_preds_b, val_logprobs_b = logprob_inference(
    model_b_best, proc_b, final_val_df, ANSWER_TOKEN_IDS, return_logprobs=True
)

correct_b = sum(p == g for p, g in zip(val_preds_b, val_gold))
val_accuracy_b = correct_b / len(val_gold)
print(f"Model B Val Accuracy: {correct_b}/{len(val_gold)} = {val_accuracy_b:.4f}")

# Fallback guard for Model B
if val_accuracy_b < VAL_ACCURACY_FALLBACK_THRESHOLD:
    print(f"FALLBACK GUARD: Model B accuracy ({val_accuracy_b:.4f}) < {VAL_ACCURACY_FALLBACK_THRESHOLD}")
    print("Do NOT include Model B in ensemble. Use single-model TTA submission instead.")
```

### Optional Cell C: Ensemble via Logprob Averaging

```python
def logprob_ensemble(logprobs_a: List[dict], logprobs_b: List[dict]) -> List[str]:
    """Average logprobs from two models and pick the best answer."""
    predictions = []
    for lp_a, lp_b in zip(logprobs_a, logprobs_b):
        avg = {k: (lp_a[k] + lp_b[k]) / 2.0 for k in lp_a}
        predictions.append(max(avg, key=avg.get))
    return predictions

# Re-run Model A inference with logprobs for val set
val_preds_a2, val_logprobs_a = logprob_inference(
    # Note: need to reload Model A here
    # ... (reload model_a_best)
    model_a_reloaded, proc_a_reloaded, final_val_df, ANSWER_TOKEN_IDS, return_logprobs=True
)

ens_val = logprob_ensemble(val_logprobs_a, val_logprobs_b)
correct_ens = sum(p == g for p, g in zip(ens_val, val_gold))
print(f"\nEnsemble Val Accuracy: {correct_ens}/{len(val_gold)} = {correct_ens/len(val_gold):.4f}")
print(f"  Model A: {val_accuracy_a:.4f}")
print(f"  Model B: {val_accuracy_b:.4f}")
print(f"  Ensemble: {correct_ens/len(val_gold):.4f}")
```

### Optional Cell D: Ensemble Test Inference + Submission

```python
# Run both models on test set with logprobs, then average
# ... (similar pattern: load each model, run logprob_inference with return_logprobs=True, then logprob_ensemble)

test_preds_ensemble = logprob_ensemble(test_logprobs_a, test_logprobs_b)

sub_ensemble = pd.DataFrame({"id": test_df["id"], "answer": test_preds_ensemble})
sub_ensemble.to_csv("submission_ensemble.csv", index=False)
print("Saved submission_ensemble.csv")

# If ensemble is better than TTA on val set, use it as final
# Otherwise keep TTA submission
```

---

## Summary of Improvements Over Baseline (0.87)

| Change | Expected Impact | Rationale |
|--------|----------------|-----------|
| 672px (from 384px) | +3-4% | 3.1x more pixels; critical for counting (51.6% of questions) |
| Domain Korean prompt (from generic English) | +1-2% | Model understands Korean recycling context natively |
| bf16 full (from 4-bit NF4) | +1-2% | No quantization error; H100 has enough VRAM |
| LoRA r=32 (from r=16) | +0.5-1% | More adapter capacity for domain adaptation |
| Dev data filtered (from train-only) | +0.5-1% | ~1,347 more training rows with quality filter (corrected from earlier overestimate) |
| Label masking | +0.5-1% | Gradient signal focused on answer prediction |
| Logprob inference (from generate+parse) | +0.5-1% | Eliminates default-to-"a" parse failures |
| Cosine schedule (from linear) | +0.3-0.5% | Better late-training convergence |
| Stratified split (from sequential) | +0.3-0.5% | Balanced validation across question types |
| TTA answer-choice shuffling (4 permutations) | +1-2% | Eliminates position bias; logprob averaging reduces variance |
| Early stopping (patience=1) | time savings | Prevents overfitting; saves 2-4h if model converges early |
| **Cumulative (conservative)** | **+6-8%** | **0.87 + 0.06 = 0.93 target** |

---

## Fallback Plan

If accuracy is below 0.93 after TTA:
1. **Train Model B** (optional phase): Different seed/LR/LoRA rank, then logprob-average ensemble
2. **Increase IMAGE_SIZE to 784**: More resolution at cost of ~50% more training time
3. **Train third model**: LoRA r=16 with lr=1e-5 for 5 epochs (longer, gentler training)
4. **TTA with more permutations**: Use all 24 permutations of a/b/c/d instead of 4

---

## Final Success Criteria

- [ ] Single notebook runs end-to-end on H100 80GB without manual intervention
- [ ] Training completes within 8 hours (leaving margin for TTA + reruns)
- [ ] GPU memory probe confirms peak < 70GB after step 1
- [ ] Timing checkpoint confirms per-step < 2.5s after step 100
- [ ] Early stopping triggers appropriately (or all epochs complete)
- [ ] Validation accuracy of Model A >= 0.87 (fallback guard passes)
- [ ] TTA inference completes within 3 hours
- [ ] submission.csv has exactly 5,074 rows with valid format
- [ ] Three submission CSVs are generated (single model, TTA, final)
