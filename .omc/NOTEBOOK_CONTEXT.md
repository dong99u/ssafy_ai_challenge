# SSAFY VQA 노트북 개발 컨텍스트

> 마지막 업데이트: 2026-04-03
> 작성: Claude Opus 4.6 (독립 분석 기반)

---

## 1. 프로젝트 개요

- **대회**: SSAFY 재활용품 VQA (Visual Question Answering)
- **과제**: 재활용품 이미지 + 질문 → 4지선다(a/b/c/d) 답변
- **질문 유형 분포** (train.csv 5073행 기준):
  - count 34.5% | material 31.0% | object 17.7% | location 10.7% | color 5.2% | ocr 0.9%
- **환경**: Google Colab A100 80GB (세션마다 40GB가 배정될 수도 있음)
- **모델**: `Qwen/Qwen3.5-9B` (BF16, 19.3GB, LoRA r=64)

---

## 2. 노트북 파일 목록

| 파일 | 설명 | 상태 |
|------|------|------|
| `ssafy_vqa_qwen3vl_h100_topscore.ipynb` | 원본 노트북 | 보존 (수정 X) |
| `ssafy_vqa_qwen3vl_h100_topscore_claude.ipynb` | Claude 수정본 (7 Fix + 3 Bug Fix + 세션 복구) | **현재 사용 중** |
| `ssafy_vqa_qwen35_h100_critic_fixed_final.ipynb` | ChatGPT 수정본 (critic 반영) | 참고용 (adapter 이름 비호환) |

---

## 3. Claude 수정본 (`_claude.ipynb`) 적용된 수정사항

### 7 Fixes (원본 대비)

| Fix | 심각도 | 셀 | 내용 |
|-----|--------|-----|------|
| 1 | P0 CRASH | Cell 5 | `train_df["answer"].isin(["a","b","c","d"])` 필터 — NaN 1건 제거 |
| 2 | P1 HIGH | Cell 7 | text prior를 `official_train_core`로만 학습 (validation leakage 제거) |
| 3 | P1 HIGH | Cell 10 | `hash()` → `hashlib.md5()` (세션 간 결정적 해시) |
| 4 | P1 HIGH | Cell 8,10 | `safe_apply_chat_template` 래퍼 (enable_thinking=False, lazy detection + fallback) |
| 5 | P2 MED | Cell 10 | predict_dataframe에 branch probs 저장 + optimize_ensemble_weights 자동 호출 |
| 6 | P2 MED | Cell 7 | template_aware_group_split에 qtype 분포 체크 + 재시도 (max 10회, skew < 5%) |
| 7 | P2 MED | Cell 8 | MCVQADataset에 set_epoch() — epoch마다 다른 옵션 셔플 |

### 3 Trace Bug Fixes (추가)

| Bug | 내용 |
|-----|------|
| BUG 1 | 미사용 branch에 `[0.25]*4` 대신 `None` 저장 + optimize에서 `x is not None` 가드 |
| BUG 2 | `Qwen/Qwen3.5-9B` 모델 존재 확인됨 (19.3GB) — 비이슈 |
| BUG 3 | `safe_apply_chat_template` — enable_thinking 미지원 시 자동 fallback |

### 세션 복구 로직 (Cell 9)

- shared_adapter가 Google Drive에 백업되어 있으면 복사만 수행 (~10초)
- 없으면 처음부터 학습 (fallback)
- count_adapter, material_adapter만 새로 학습

---

## 4. 학습 진행 상태

### 완료

- **shared_adapter**: 2 epoch 완료, valid_loss=0.02063
  - 백업: `/content/drive/MyDrive/vqa_ckpt/shared_adapter/best`
  - 학습 시간: ~4시간

### 미완료

- **count_adapter**: epoch 1의 27%에서 세션 끊김 (A100 40GB OOM도 발생)
- **material_adapter**: 시작 안 됨

### 재시작 방법

1. Colab에서 GPU를 **A100 80GB**로 설정
2. `_claude.ipynb` 업로드
3. Cell 1~8 순서대로 실행
4. Cell 9 실행 → shared_adapter Drive 복구 + count/material 학습
5. Cell 10~11 → 추론 + 제출

### A100 40GB 배정 시

Cell 8과 Cell 9 사이에 아래 셀 추가 실행:
```python
CFG.per_device_batch_size = 1
CFG.grad_accum_steps = 16
CFG.train_max_pixels = 1024 * 1024
import os; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc, torch; gc.collect(); torch.cuda.empty_cache()
```

---

## 5. ChatGPT 수정본 Critic 요약

### ChatGPT가 잘한 것 (cherry-pick 가치 있음)

| 항목 | 설명 | 채택 가치 |
|------|------|-----------|
| StratifiedGroupKFold | 우리 retry보다 수학적으로 정확한 stratified split | 높음 |
| reblend_prediction_dataframe | VLM 재추론 없이 가중치만 바꿔서 재평가 | 높음 |
| pseudo threshold proxy tuning | text prior proxy로 dev threshold grid search | 조건부 |
| 24-type weight routing | qtype_detail별 가중치 최적화 | 조건부 |
| flash-attn 이중 fallback | install + model load 양쪽에서 SDPA fallback | 높음 |
| retrain_final_after_tune | holdout 튜닝 → weight 최적화 → full retrain 워크플로우 | 높음 |
| TrashNet material+object 이중 생성 | 같은 이미지로 재질/물체 두 가지 QA | 합리적 |

### ChatGPT가 놓친 것

| 항목 | 심각도 | 설명 |
|------|--------|------|
| enable_thinking=False 미적용 | **심각** | prompt 지시만으로는 <think> 토큰 생성 완전 차단 불가. logprob 위치 계산 오류 위험 |
| epoch-varying shuffle 없음 | 중간 | 매 epoch 동일 셔플 → 옵션 셔플 augmentation 효과 반감 |
| 세션 복구 로직 없음 | 중간 | Colab 끊기면 처음부터 재학습 |
| adapter 이름 비호환 | **치명적** (현재 상황) | `stage1_holdout_shared_adapter` vs `shared_adapter` — Drive 백업과 호환 안 됨 |

### 설계 의문점

| 항목 | ChatGPT 선택 | 판단 |
|------|-------------|------|
| `use_fulltext_logprob=False` 기본값 | letter-only가 학습과 일관적이라고 주장 | A/B 실험 없이 변경은 성급 |
| 24-type detail weights (min=16) | 16샘플로 가중치 최적화 | 오버피팅 위험 높음, 32+ 권장 |
| Dirichlet 512 trial | random search 방식 | Nelder-Mead보다 탐색 넓지만 정밀도 낮음, 실전 차이는 작을 것 |
| pseudo threshold proxy | text prior accuracy로 threshold 선택 | 약한 proxy지만 고정값보다 나음 |

---

## 6. 알려진 잔존 이슈 (Trace 분석)

| 이슈 | 심각도 | 상태 |
|------|--------|------|
| prefix_len single vs batch 불일치 | Moderate (live 검증 필요) | 미검증 — 같은 이미지면 보통 동일하지만 확인 필요 |
| bare letter 연결 tokenization | Moderate | `prefix_text + "a"` BPE 병합 위험 — tokenizer 의존 |
| asdict(CFG) 동적 속성 누락 | Low | `attn_implementation`이 저장 config에서 빠짐 |

---

## 7. 전략 메모

### 모델 크기 스케일링 옵션

다른 팀이 **큰 모델 + baseline 코드**로 0.9 달성했다는 정보:

| 모델 | VRAM (BF16) | VRAM (4-bit) | A100 80GB 학습? |
|------|-------------|-------------|----------------|
| Qwen2.5-VL-7B | ~14GB | ~4GB | O (여유) |
| Qwen3.5-9B (현재) | ~19GB | ~5GB | O |
| Qwen2.5-VL-72B | ~144GB | ~40GB | **4-bit + LoRA = ~55-65GB → 가능** |

**판단 기준**: 현재 9B 노트북의 holdout accuracy를 먼저 확인:
- 0.85 이하 → 72B 전환 ROI 높음
- 0.88+ → 현재 파이프라인 미세 튜닝이 나을 수 있음

### flash-attn 상태

- pre-built wheel 설치 완료: `flash_attn-2.8.3+cu128torch2.10-cp312`
- `CFG.attn_implementation = "flash_attention_2"` 설정됨
- 단, "fast path not available" 로그가 뜸 → flash-linear-attention 별도 설치 필요할 수 있음 (성능 영향은 미미)
