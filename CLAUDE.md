# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SSAFY 15기 AI Challenge: **Recycling image VQA (Visual Question Answering)**. Given an image of recyclable items and a Korean-language multiple-choice question (a/b/c/d), predict the correct answer. Evaluation metric is **Accuracy**. Submission is a CSV with columns `id,answer`.

## Environment Setup

- **Python**: 3.12 (pinned in `.python-version`)
- **Package manager**: `uv` (not pip)
- **Install (macOS/CPU)**: `uv sync --extra cpu`
- **Install (CUDA)**: `uv sync --extra cu130`
- **Add a package**: `uv add <pkg> && uv sync --extra cpu` (or `--extra cu130`)
- **Activate venv**: `source .venv/bin/activate`
- **Verify torch**: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

PyTorch extras (`cpu` vs `cu130`) are mutually exclusive. Never `uv add` torch/torchvision/torchaudio directly; edit `pyproject.toml` instead.

## Data Layout

Data lives in `data/` (gitignored). Expected structure:

```
data/
  csv/
    train.csv      # 5,073 rows: id, path, question, a, b, c, d, answer
    test.csv       # 5,074 rows: id, path, question, a, b, c, d (no answer)
    dev.csv        # 4,413 rows: id, path, question, a, b, c, d, answer1-5
    sample_submission.csv
  train/           # ~5,074 JPG images
  test/            # ~5,075 JPG images
  dev/             # ~4,414 JPG images
```

- `dev.csv` has 5 annotator responses (`answer1`-`answer5`) instead of a single gold `answer` — useful for pseudo-labeling or majority-vote strategies.
- Image paths in CSVs are relative (e.g., `train/train_0001.jpg`). Notebooks resolve them relative to the CSV directory or via absolute paths.

## Baseline Architecture

Two reference notebooks in `baseline/`:
- `(260324)_baseline_colab.ipynb` — Colab (T4 GPU)
- `(260325)_baseline_desktop5060ti.ipynb` — Local desktop (RTX 5060 Ti)

Both follow the same pipeline:

1. **Model**: `Qwen/Qwen2.5-VL-3B-Instruct` (vision-language model)
2. **Quantization**: 4-bit NF4 via `BitsAndBytesConfig`
3. **Fine-tuning**: LoRA (r=8, alpha=16) on attention + MLP projections via PEFT
4. **Training**: AdamW + linear warmup scheduler, gradient accumulation=4, mixed precision (bfloat16)
5. **Inference**: `model.generate(max_new_tokens=2, do_sample=False)` → parse single letter `a/b/c/d`
6. **Prompt format**: Qwen chat template with system instruction + image + Korean MC question

Key classes/functions shared by both notebooks:
- `VQAMCDataset(Dataset)` — loads image + builds chat messages
- `DataCollator` — applies processor chat template + tokenization
- `build_mc_prompt()` — formats question with choices
- `extract_choice()` — parses model output to a/b/c/d (defaults to "a" on failure)

## Competition Rules

- **Team**: 5-person team, daily submission limit of 20
- **Models**: HuggingFace pretrained models only; API calls for inference are forbidden
- **Data**: External public data and augmentation are allowed; data leakage from test set is forbidden
- **Techniques**: LoRA, quantization, prompt engineering, data augmentation all permitted
- **Reproducibility**: Submitted code must reproduce submitted results

## Harness Protocol

### 상태 관리
- 세션 시작 시 `claude-progress.txt`가 있으면 반드시 먼저 읽어라
- 작업 단위 완료마다 `claude-progress.txt`를 업데이트하라
  - 형식: `## 완료` / `## 진행 중` / `## 남은 것` / `## 이슈`
- 실험 결과(모델, 하이퍼파라미터, accuracy 등)는 `experiments.json`에 JSON 배열로 누적 기록하라

### 커밋 규율
- 하나의 논리적 작업 단위마다 git commit
- 실험 결과가 나오면 즉시 commit (롤백 포인트 확보)

### 핸드오프
- 큰 작업 완료 시 `claude-progress.txt`에 다음 세션이 즉시 이해할 수 있도록 기록하라
  - 무엇을 했는지, 무엇이 남았는지, 알려진 이슈는 무엇인지
