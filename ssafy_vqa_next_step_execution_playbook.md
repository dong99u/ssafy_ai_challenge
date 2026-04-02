# SSAFY 재활용품 VQA 다음 단계 실행 플랜

이 문서는 지금까지 정리한 전략을 **실제로 점수로 바꾸기 위한 실행용 플랜**이다.  
목표는 `빠르게 안정적인 제출본 확보 → 고점 실험 → 리더보드 기반 최종 앙상블 고정` 순서로 움직이는 것이다.

---

## 1. 지금 바로 내릴 결론

### 실행 우선순위
1. **실전 1차 주력은 `Qwen3-VL-8B-Instruct`**로 간다.
   - 이유: 이미 준비된 Colab H100 / 로컬 CUDA 스크립트가 있고 구현 리스크가 낮다.
   - 남은 시간이 짧을수록, **안정적으로 여러 실험을 돌릴 수 있는 모델**이 더 유리하다.
2. **여유가 있을 때만 `Qwen3.5-9B` 추가 실험**을 붙인다.
   - ceiling은 더 높을 수 있지만, 설치/속도/메모리/세부 튜닝 리스크가 있다.
3. 최종 제출은 가능하면 아래 3종을 관리한다.
   - **Conservative**: shared + count + text prior
   - **Balanced**: Conservative + option permutation + heavy second pass
   - **Aggressive**: Balanced + external data 강하게 반영 + text prior 비중 상향

### 가장 중요한 운영 원칙
- **384x384 고정 stretch 금지**
- **free generation 금지, logprob reranking 사용**
- **dev는 shared 전체모델보다 count expert 중심으로 사용**
- **public LB만 따라가지 말고 OOD 위험 q_type을 따로 관리**

---

## 2. 무엇을 먼저 돌릴까

현재 준비된 파일 기준으로 실행 순서는 아래가 가장 좋다.

### Colab H100
- 우선 실행 파일: `ssafy_vqa_qwen3vl_colab_h100.ipynb`
- 코드형 실행 파일: `ssafy_vqa_qwen3vl_colab_h100.py`

### 로컬 CUDA 13.0
- 우선 실행 파일: `ssafy_vqa_qwen3vl_local_cu130.ipynb`
- 코드형 실행 파일: `ssafy_vqa_qwen3vl_local_cu130.py`

### 추천 역할 분담
- **Colab H100**: 실제 학습, 최종 고해상도 추론, heavy second pass
- **로컬 CUDA**: config 실험, text prior/EDA, lightweight validation, submission 비교

---

## 3. 실험 트랙 전체 그림

### Track A: 안전한 고득점 확보
가장 먼저 끝내야 하는 트랙이다.

1. train만으로 shared adapter 학습
2. count expert 추가 학습
3. text prior 결합
4. option permutation averaging
5. submission 1차 생성

이 트랙만 제대로 돌아도, baseline 대비 큰 개선이 기대된다.

### Track B: 점수 밀어올리기
Track A가 안정적으로 돌아간 뒤 붙인다.

1. dev pseudo label을 **count 중심**으로 추가
2. TrashNet synthetic material QA 추가
3. OCR boost / easyocr 추가
4. uncertain-only heavy second pass 추가
5. blend weights 재조정

### Track C: 최고점용 공격 실험
시간이 남으면 붙인다.

1. TACO 수동 연결
2. AI Hub synthetic QA 추가
3. Qwen3.5-9B 보조 모델 추가
4. model ensemble

---

## 4. 실험 우선순위표

아래의 기대 효과는 이 대회 구조를 바탕으로 한 **실전 감각적 우선순위**다. 절대값이 아니라 `무엇부터 해야 하는지`를 위한 순서다.

| 우선순위 | 실험 | 목적 | 구현 난도 | 시간 | 기대 효과 | 바로 실행 여부 |
|---|---|---|---:|---:|---:|---|
| P0-1 | Qwen3-VL-8B shared 학습 | 기본 성능 확보 | 중 | 중 | 매우 큼 | 필수 |
| P0-2 | count expert 학습 | count 비중 대응 | 중 | 중 | 큼 | 필수 |
| P0-3 | logprob reranking | 4지선다 안정화 | 낮음 | 낮음 | 큼 | 이미 반영, 필수 |
| P0-4 | option permutation averaging | 선택지 위치 편향 감소 | 낮음 | 낮음 | 중간 | 필수 |
| P0-5 | text prior 결합 | material/object 계열 보강 | 낮음 | 낮음 | 중간~큼 | 필수 |
| P1-1 | dev pseudo(count 위주) | count expert 강화 | 낮음 | 낮음 | 중간 | 강추 |
| P1-2 | TrashNet synthetic QA | material_general 강화 | 낮음 | 낮음 | 중간~큼 | 강추 |
| P1-3 | OCR boost | brand/문자/용량 대응 | 중 | 낮음 | 작음~중간 | 추천 |
| P1-4 | heavy second pass | 불확실 샘플 재판정 | 중 | 중 | 중간 | 추천 |
| P1-5 | q_type별 blend 재조정 | 분포 대응 | 낮음 | 낮음 | 중간 | 추천 |
| P2-1 | TACO 연결 | count/object OOD 강화 | 중 | 중 | 중간 | 여유 있으면 |
| P2-2 | AI Hub synthetic QA | material/object ceiling 상승 | 높음 | 높음 | 큼 | 여유 있으면 |
| P2-3 | Qwen3.5-9B 추가 | 최고 ceiling 탐색 | 높음 | 높음 | 중간~큼 | 마지막 |

---

## 5. 실험 순서 상세

## Step 0. 데이터/환경 sanity check
가장 먼저 아래만 확인한다.

- `cfg.data_root` 경로 정상 여부
- train / dev / test csv 정상 로드 여부
- 이미지 path 실제 존재 여부
- 1 batch forward 정상 여부
- submission.csv 생성 여부

### 성공 기준
- 에러 없이 1개 sample 학습/추론 통과
- debug csv 생성
- memory OOM 없음

---

## Step 1. 안전한 1차 제출본 확보

### 설정
아래 설정으로 첫 학습/추론을 만든다.

```python
cfg.use_dev_pseudo = False
cfg.use_trashnet = False
cfg.use_taco_manual_hook = False
cfg.do_train_shared = True
cfg.do_train_count_expert = True
cfg.do_heavy_second_pass = False
cfg.use_easyocr = False
cfg.infer_num_permutations = 2
```

### 목적
- 외부 요인 없이 **학습 파이프라인이 실전 제출까지 완주**되는지 본다.
- 가장 먼저 **안정적인 기준 submission**을 확보한다.

### 평가 포인트
- 전체 CV
- count / material / object 세 축 별도 점검
- 추론 시간
- OOM 여부

### 통과 기준
- baseline 대비 뚜렷한 개선
- count 쪽이 baseline보다 좋아야 함
- material/object가 text-only보다 납득 가능한 상승

---

## Step 2. text prior + permutation 적용

### 설정
```python
cfg.use_text_prior = True
cfg.infer_num_permutations = 4
```

### 목적
- material_general / object_type / recycle_class 쪽 점수 상승
- 선택지 순서 편향 감소

### 운영 팁
- permutation은 2에서 시작하고, 시간이 허용하면 4로 올린다.
- public LB가 올라가더라도, **count가 떨어지는지** 꼭 확인한다.

### 통과 기준
- CV 또는 holdout에서 material/object 개선
- count 성능 급락 없음

---

## Step 3. dev pseudo는 count expert에만 강하게 사용

Discussion과 EDA를 반영하면 dev는 train/test와 달리 count-heavy다.  
따라서 shared 전체 모델에 많이 섞기보다 **count expert용 약한 pseudo label**로 쓰는 게 더 낫다.

### 설정
```python
cfg.use_dev_pseudo = True
cfg.pseudo_min_conf_shared = 0.85
cfg.pseudo_min_conf_count = 0.60
```

### 추천 운영
- shared: `0.85` 이상만 아주 제한적으로
- count expert: `0.60` 이상 사용
- 다수결이 약한 샘플은 버린다.

### 통과 기준
- count 관련 CV 또는 holdout 개선
- material/object 악화 없음

---

## Step 4. TrashNet synthetic QA 추가

이 단계는 discussion에서 강조된 **material_general 강화**와 직접 연결된다.

### 설정
```python
cfg.use_trashnet = True
```

### 목적
- glass / metal / paper / plastic 혼동 감소
- material_general / material_bottle / material_cup 강화

### 운영 팁
- 외부데이터는 처음부터 너무 많이 섞지 않는다.
- `train : external = 1 : 0.2` 느낌으로 시작하고, 성능을 보고 늘린다.
- synthetic question 템플릿은 train 문체와 최대한 비슷하게 만든다.

### 통과 기준
- material_general 상승
- object_type이 무너지지 않음

---

## Step 5. OCR boost + heavy second pass

이 단계는 전체 비중은 작지만, 리더보드에서 미세한 차이를 만드는 단계다.

### 설정
```python
cfg.use_easyocr = True
cfg.do_heavy_second_pass = True
cfg.heavy_second_pass_margin = 0.25
cfg.heavy_second_pass_qtypes = ("ocr_text", "location")
```

### 목적
- OCR / brand / volume / location 계열 보강
- 애매한 샘플만 고해상도 재판정

### 운영 팁
- heavy second pass는 **top1-top2 margin이 작은 샘플만** 대상으로 한다.
- 모든 샘플에 적용하면 시간 대비 이득이 작다.

### 통과 기준
- 전체 점수 유지 또는 소폭 상승
- OCR 계열 정답률 상승
- 추론 시간이 감당 가능 수준

---

## Step 6. q_type별 blend weights 재조정

현재 스크립트의 기본 가중치는 좋은 시작점이지만, discussion/EDA를 반영하면 아래 방향으로 조정 실험을 하는 게 좋다.

### 기본 원칙
- **count**: text 비중을 낮추고 count expert 비중을 높인다.
- **material/object**: shared + text prior를 함께 쓴다.
- **location/other**: 과도한 text prior 의존을 줄인다.
- **ocr_text**: OCR boost 가중치를 따로 둔다.

### 추천 시작점
```python
cfg.blend_weights = {
    "count":    {"shared": 0.55, "count": 0.33, "text": 0.12, "ocr": 0.00},
    "material": {"shared": 0.70, "count": 0.00, "text": 0.30, "ocr": 0.00},
    "object":   {"shared": 0.68, "count": 0.10, "text": 0.22, "ocr": 0.00},
    "color":    {"shared": 0.78, "count": 0.00, "text": 0.22, "ocr": 0.00},
    "location": {"shared": 0.88, "count": 0.00, "text": 0.07, "ocr": 0.05},
    "ocr_text": {"shared": 0.55, "count": 0.00, "text": 0.05, "ocr": 0.40},
}
```

---

## 6.5. risk-aware validation을 반드시 추가

업로드된 EDA 이미지를 보면 `material`, `location`, `other`는 텍스트 편향과 OOD 위험이 함께 보인다.  
이건 **일반 CV만 보면 성능을 과대평가할 수 있다**는 뜻이다.

그래서 검증은 최소 2개로 본다.

### 검증 A: 일반 fold
- 현재 점수 추세 확인용
- 빠른 실험 필터링용

### 검증 B: 질문 템플릿/문장 기준 group split
- exact question 또는 normalized question을 group으로 묶고 분리
- material/location/other의 과대평가를 걸러내는 용도

### 운영 원칙
- public LB가 올라가도, group split 성능이 급락하면 위험 실험으로 표시
- 최종 앙상블은 **일반 CV + group split + public LB** 셋을 같이 보고 선택

---

## 7. 추천 제출 세트

최종적으로는 한 개만 제출하지 말고, 최소 3개 버전을 유지하는 게 좋다.

### 제출 A: Conservative
- train only 또는 train + 매우 제한된 pseudo
- shared + count + text prior
- permutation 2
- heavy second pass off

**용도**: 가장 안전한 기준 제출본

### 제출 B: Balanced
- train + count pseudo + TrashNet
- shared + count + text prior + permutation 4
- heavy second pass on(불확실 샘플만)

**용도**: 기본 최종 후보

### 제출 C: Aggressive
- train + pseudo + TrashNet + TACO 또는 AI Hub synthetic
- text prior 비중 소폭 상향
- OCR boost on
- heavy second pass on

**용도**: public LB 고점 탐색용

---

## 8. public LB 해석 규칙

리더보드에 너무 쉽게 끌려가면 안 된다. 아래 규칙으로 관리한다.

### 신뢰도 높음
- public LB 상승
- 일반 CV 상승
- group split 성능 유지 또는 상승

### 신뢰도 중간
- public LB 상승
- 일반 CV 보합
- group split 약간 하락

### 위험
- public LB 상승
- 일반 CV 하락
- group split 크게 하락

이 경우는 **텍스트 편향 과적합**일 가능성이 높다.

---

## 9. Colab H100 / 로컬 CUDA 운용 분리

## Colab H100에서 할 일
- shared adapter 학습
- count expert 학습
- permutation 4 추론
- heavy second pass
- 최종 submit 생성

### H100 추천 값
```python
cfg.use_bf16 = True
cfg.train_load_in_4bit = True
cfg.infer_load_in_4bit = False
cfg.shared_train_batch_size = 2
cfg.count_train_batch_size = 2
cfg.gradient_accumulation_steps = 8
cfg.train_max_side = 1344
cfg.infer_max_side = 1536
cfg.heavy_max_side = 1792
```

## 로컬 CUDA 13에서 할 일
- lightweight 실험
- text prior / rule tuning
- 빠른 디버깅
- submission 비교

### 로컬 추천 값
```python
cfg.use_bf16 = True
cfg.train_load_in_4bit = True
cfg.infer_load_in_4bit = True
cfg.shared_train_batch_size = 1
cfg.count_train_batch_size = 1
cfg.gradient_accumulation_steps = 16
cfg.train_max_side = 1120
cfg.infer_max_side = 1344
cfg.heavy_max_side = 1536
```

---

## 10. OOM / 속도 문제 발생 시 fallback 순서

문제가 생기면 아래 순서로 줄인다.

1. `infer_num_permutations: 4 -> 2`
2. `heavy second pass` 끄기
3. `heavy_max_side` 줄이기
4. `infer_load_in_4bit = True`
5. `train_max_side / infer_max_side` 줄이기
6. batch size 1로 낮추기

절대 마지막까지 유지해야 하는 것은 아래다.
- answer-only masking
- logprob reranking
- option shuffle augmentation
- q_type 라우팅

---

## 11. 가장 현실적인 24시간 플랜

시간이 정말 부족하면 이 순서로 간다.

### 0~2시간
- 환경 확인
- train only shared + count 학습
- 첫 submission 생성

### 2~6시간
- text prior + permutation 적용
- Balanced 제출 생성

### 6~10시간
- dev pseudo(count) 추가
- TrashNet 추가
- 재학습 / 재추론

### 10~16시간
- OCR boost
- heavy second pass
- blend weights 조정

### 16~24시간
- TACO 또는 AI Hub synthetic 연결
- Aggressive 제출 생성
- 최종 3개 후보 비교 후 고정

---

## 12. 실험 중단 / 채택 기준

### 바로 중단할 실험
- 학습/추론 안정성이 낮아 반복 재현이 안 되는 경우
- public LB만 오르고 group split이 크게 망가지는 경우
- count가 눈에 띄게 무너지는 경우

### 채택할 실험
- public LB, 일반 CV, group split 중 2개 이상이 좋아지는 경우
- material/object가 좋아지면서 count가 유지되는 경우
- 전체 점수는 비슷해도 OOD q_type이 좋아지는 경우

---

## 13. 마지막 추천

지금 시점에서 가장 좋은 실행 순서는 아래다.

1. **Qwen3-VL-8B로 안전한 1차 제출본 확보**
2. **text prior + permutation 적용**
3. **dev pseudo(count) + TrashNet 추가**
4. **Balanced 제출 생성**
5. **OCR/heavy second pass로 미세 조정**
6. **여유 있으면 TACO / AI Hub / Qwen3.5-9B 추가**

핵심은 하나다.  
**가장 강한 모델 하나를 찾는 것보다, 이 대회에 맞는 라우팅/스코어링/외부데이터 조합을 빨리 안정화하는 쪽이 더 점수로 이어질 가능성이 높다.**
