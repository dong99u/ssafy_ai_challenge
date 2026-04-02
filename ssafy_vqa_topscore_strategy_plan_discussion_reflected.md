# SSAFY 재활용품 VQA 최고 점수 전략 (Discussion 반영판)

작성 목적: 대회 Discussion에서 공유된 인사이트와 업로드된 train/dev/test 분석 결과를 반영해, **리더보드 최고 점수**를 목표로 한 실전 전략을 정리한다.

---

## 1. 한 줄 결론

가장 승률이 높은 방향은 아래 조합이다.

1. **메인 모델**
   - **1순위 실험 트랙:** `Qwen/Qwen3.5-9B`
   - **안정형/백업 트랙:** `Qwen/Qwen3-VL-8B-Instruct`
   - 최종 제출은 가능하면 **두 모델 앙상블**

2. **학습 전략**
   - 기본 train만으로 끝내지 말고,
   - **AI Hub 생활 폐기물 / 재활용품 분류 및 선별 데이터 + TACO + TrashNet**을
   - 대회 포맷과 동일한 **4지선다 VQA synthetic data**로 변환해 추가 학습

3. **전문가 분업**
   - 하나의 범용 모델만 쓰지 말고,
   - `shared` + `material/object expert` + `count expert` 구조로 간다.

4. **추론 전략**
   - a/b/c/d 생성이 아니라 **선택지 logprob reranking**
   - **option permutation averaging**
   - **text prior + OCR + count 보조 feature** 합성

5. **가장 중요한 전처리**
   - **384x384 강제 stretch 금지**
   - **원본 종횡비 유지 + dynamic resolution + 필요 시 crop**

---

## 2. Discussion 반영 후 바뀌는 핵심 판단

### 2.1 Train/Test에서 가장 큰 비중은 material_general + object_type
Discussion에서 공유된 분포와 실제 CSV를 다시 보면, train/test의 주력 문제는 count만이 아니라 **소재(material)**와 **객체 종류(object type)**다.

내가 업로드된 CSV를 다시 분류해 본 대략적 비중:

### broad q_type 비중
| split | count | material | object_id | color | recycle_class | location | brand_product | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 34.4 | 31.8 | 16.7 | 7.3 | 4.1 | 2.8 | 1.1 | 1.9 |
| dev | 71.4 | 13.0 | 6.7 | 2.9 | 1.3 | 2.9 | 0.4 | 1.4 |
| test | 33.8 | 30.1 | 18.8 | 7.2 | 4.4 | 2.6 | 1.0 | 2.0 |

### detail q_type 상위 비중
| split | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| train | material_general 23.1 | object_type 16.7 | count_bottle 9.6 | count_general 7.5 | count_cup 6.7 |
| dev | count_general 21.2 | count_bottle 18.6 | count_box 13.5 | count_cup 11.6 | material_general 8.7 |
| test | material_general 21.9 | object_type 18.8 | count_bottle 9.8 | count_general 8.1 | count_cup 6.6 |

즉, **dev는 count-heavy라 train/test와 분포가 다르다.**  
따라서 dev를 “전체 모델”에 강하게 섞는 것은 오히려 해가 될 수 있다.  
**dev는 count expert 강화용 약한 pseudo label 소스**로 보는 게 맞다.

---

## 3. EDA와 Discussion에서 얻는 실전 시사점

### 3.1 384x384 정사각형 stretch는 손해
Discussion에서 공유된 내용대로 원본 이미지가 세로형 비중이 높다.  
이 대회 이미지는 휴대폰 사진 특성상 **세로로 길고, 작은 물체/라벨/뚜껑/빨대/병목부** 같은 국소 정보가 중요하다.

**따라서 금지할 것**
- 무조건 384x384 resize
- center crop만 사용하는 단일 뷰 추론
- 작은 물체를 날리는 aggressive compression

**권장할 것**
- longest-side 기준 resize 후 pad
- dynamic resolution
- 고해상도 1뷰 + 객체 중심 crop 1뷰
- count / OCR 샘플만 2차 high-res pass

---

### 3.2 텍스트 편향은 생각보다 강하다
Discussion의 “가장 긴 보기 선택만으로 맞는 정확도”, “텍스트 누수”, “OOD 비율”은 매우 중요한 힌트다.

내가 train으로 빠르게 계산해 본 **가장 긴 보기 하나만 고르는 naive baseline**은 다음과 비슷했다.

| q_type | naive acc(%) |
|---|---:|
| material | 56.5 |
| recycle_class | 55.5 |
| object_id | 51.2 |
| location | 46.4 |
| brand_product | 35.8 |
| color | 28.3 |
| count | 26.2 |

이 뜻은 간단하다.

- material/object/recycle/location 일부는 **이미지 없이도 텍스트만으로 꽤 맞출 수 있다**
- 반대로 count는 **텍스트 편향이 약하므로 비전 정보가 더 중요하다**

그래서 최종 시스템은 아래처럼 가야 한다.

- **material/object/recycle/location**:  
  VLM score + text prior를 강하게 결합
- **count**:  
  VLM score + count expert + detector/heuristic을 강하게 결합
- **OCR/brand_product**:  
  VLM score + OCR engine match를 결합

---

### 3.3 Test의 질문 반복률은 높지만 OOD qtype은 분명히 있다
내가 업로드 CSV 기준으로 보니 **test 질문 문자열의 약 64.4%는 train에 동일 문자열이 존재**했다.  
특히 count/material/object는 반복률이 높고, brand/location/other는 낮다.

대략적인 exact question overlap:
- count: 71.7%
- material: 70.5%
- object_id: 64.4%
- color: 51.6%
- location: 27.8%
- brand_product: 17.3%
- other: 26.9%

따라서:
- count/material/object는 **질문 템플릿 학습**이 잘 먹힌다.
- location/brand/other는 **OOD 대응**이 중요하다.
- 즉, **shared 한 개만으로 밀지 말고 q_type 라우팅**이 필요하다.

---

## 4. 최종 모델 전략

## 4.1 메인 추천 모델
### A트랙: 최고 ceiling
- **Qwen3.5-9B**
- 이유:
  - 최신 공식 라인업
  - Qwen 공식 설명상 **Qwen3-VL 대비 시각 이해까지 포함한 벤치에서 우위**
  - H100 1장 기준 4-bit QLoRA / bf16 추론이 현실적

### B트랙: 안정형
- **Qwen3-VL-8B-Instruct**
- 이유:
  - 이미 검증된 VLM 구조
  - 이미지 이해 + OCR + 공간 이해가 강함
  - 구현 안정성이 높고, fallback으로 좋음

### 최종 추천
- 시간 부족: **Qwen3-VL-8B 단독**
- 최고 성능 추구: **Qwen3.5-9B 주력 + Qwen3-VL-8B 앙상블**

---

## 4.2 양자화 / 학습 방식
### Colab H100
- 학습: **4-bit QLoRA**
- 추론:
  - 빠른 1차 추론은 4-bit 또는 bf16
  - 최종 submit 후보는 가능하면 bf16 재추론

### 로컬 CUDA
- 기본은 4-bit
- 로컬은 full fine-tuning보다 **adapter 실험 + validation**용으로 활용

### 권장 이유
- 4-bit QLoRA는 메모리 효율이 좋고,
- 작은/중간 규모 데이터에서 full fine-tune보다 **시간 대비 효율**이 좋다.

---

## 5. 외부 데이터 전략 (Discussion 핵심 반영)

Discussion에서 제일 가치 있는 포인트는 이것이다.

> material_general 성능을 끌어올리기 위해  
> AI Hub의 생활 폐기물 데이터를  
> 대회와 같은 "질문 + a,b,c,d" 형식으로 합성한다.

이 방향은 매우 좋고, 나는 여기서 더 나아가 **소재 + 객체 + 개수**까지 모두 synthetic VQA화하는 걸 추천한다.

### 5.1 가장 먼저 쓸 외부 데이터
1. **AI Hub 생활 폐기물**
   - 용도: material_general, object_type, recycle_class synthetic QA
   - 장점: 한국어 폐기물 taxonomy가 대회 도메인과 잘 맞음

2. **AI Hub 재활용품 분류 및 선별 데이터**
   - 용도: material_general, object_type, fine-grained subtype
   - 장점: 규모가 크고 라벨 구조가 풍부함

3. **TACO**
   - 용도: count expert, in-the-wild trash detection, clutter scene 적응
   - 장점: 실제 길거리/복잡 배경에 강함

4. **TrashNet**
   - 용도: glass / metal / paper / plastic material 분류 보조
   - 장점: 소재 구분 학습용으로 가볍고 빠름

---

### 5.2 외부 데이터를 어떻게 VQA로 변환할까
#### material_general
예:
- 질문: 사진 속 주요 재활용품의 재질은 무엇인가요?
- 보기: 유리 / 금속 / 종이 / 플라스틱
- 정답: 플라스틱

#### object_type
예:
- 질문: 사진에 보이는 재활용 가능한 물체의 종류는 무엇인가요?
- 보기: 캔 / 유리병 / 플라스틱 병 / 종이팩
- 정답: 플라스틱 병

#### recycle_class
예:
- 질문: 사진 속 물체는 어떤 분리수거 항목에 해당하나요?
- 보기: 캔류 / 병류 / 종이류 / 비닐류
- 정답: 병류

#### count
(TACO/AI Hub bbox 활용)
- 질문: 사진에 보이는 플라스틱 병은 몇 개입니까?
- 보기: 1개 / 2개 / 3개 / 4개
- 정답: 3개

---

### 5.3 가장 중요한 포인트: 전체를 다 늘리지 말고 "부족한 qtype만" 늘려라
외부 데이터는 많이 넣는다고 항상 좋아지지 않는다.

권장 우선순위:
1. **material_general**
2. **object_type**
3. **count_bottle / count_general / count_cup / count_can**
4. **recycle_class**
5. color / location / OCR는 보조적으로

즉, Discussion에서 나온 말처럼 **material_general 증강은 1순위**가 맞다.  
여기에 나는 **object_type까지 묶어서 같이 키우는 전략**을 추가 추천한다.

---

## 6. Dev 사용 전략 (아주 중요)

dev는 정답이 아니라 5명의 응답이다.  
그리고 분포도 test와 다르다.  
따라서 **dev 전체를 정답처럼 넣는 건 위험**하다.

### 6.1 내가 권장하는 방식
- 5개 응답을 vote distribution으로 변환
- `3표 이상`일 때만 pseudo label 후보로 사용
- `4표 이상`이면 높은 가중치
- `3표`면 낮은 가중치
- `2-2-1`, `2-1-1-1` 형태는 거의 버림

내가 다시 계산한 dev 합의도:
- **3표 이상 다수결:** 약 79.5%
- **4표 이상 일치:** 약 20.4%
- **5표 완전 일치:** 0%

즉, dev는 **soft label**로 써야 하고,  
특히 count 비중이 과하게 높으므로 **count expert에만 강하게 반영**하는 게 맞다.

### 6.2 dev 사용 원칙
- shared model: 약한 weight
- material/object expert: 거의 안 씀
- count expert: 적극 활용
- stacker calibration: 사용 가능

---

## 7. 학습 구조: 단일 모델이 아니라 전문가 구조

### 7.1 Shared model
역할:
- 전체 qtype 공통 기반
- material/object/count/color 전반 커버

데이터:
- train 원본
- 외부 synthetic 일부
- dev는 아주 약하게 또는 제외

---

### 7.2 Material/Object expert
역할:
- material_general, object_type, recycle_class 집중
- test의 큰 비중을 차지하는 핵심 점수원

데이터:
- train 중 해당 qtype
- AI Hub 생활 폐기물 synthetic
- AI Hub 재활용품 분류 및 선별 synthetic
- TrashNet material synthetic

가중치:
- final ensemble에서 높은 비중

---

### 7.3 Count expert
역할:
- count_bottle / count_general / count_cup / count_box / count_can

데이터:
- train count 샘플
- dev pseudo label
- TACO / AI Hub bbox 기반 synthetic count QA

추가 기능:
- detector heuristic
- crop view
- uncertainty re-check

---

### 7.4 OCR / Brand / Location 보조 expert
비중은 작지만 리더보드 상위권에서 차이를 만든다.

추천:
- 별도 대형 학습보다
- **OCR 엔진 + VLM score fusion** 형태가 시간 대비 효율이 높음

---

## 8. 프롬프트 / 학습 / 추론 설계

## 8.1 정답 생성이 아니라 reranking
하지 말 것:
- 자유 생성 후 a/b/c/d 파싱

할 것:
- 각 선택지에 대해 조건부 점수 계산
- **가장 높은 logprob 선택**

예시:
- prompt: 질문 + 선택지 + "정답:"
- candidate:
  - "a"
  - "b"
  - "c"
  - "d"
- 혹은 letter보다 더 좋게:
  - "정답은 유리"
  - "정답은 금속"
  - "정답은 종이"
  - "정답은 플라스틱"

후자가 OCR/브랜드/유형 문제에서 더 안정적인 경우가 많다.

---

## 8.2 answer-only masking
학습 loss는 **정답 토큰에만** 걸어야 한다.
- system/user/prompt/image tokens/pad는 `-100`
- 정답 문자열만 예측

baseline처럼 프롬프트 전체를 복사해 맞히게 하면 비효율적이다.

---

## 8.3 option shuffle augmentation
반드시 넣을 것:
- epoch마다 a/b/c/d 순서를 바꿈
- label은 함께 remap
- letter bias 제거에 도움

---

## 8.4 option permutation averaging
추론 시:
- 같은 샘플을 보기 순서 2~4개 permutation으로 다시 평가
- 원래 letter 위치로 환원해 평균 score
- 안정적으로 0.x ~ 1.x% 개선될 가능성이 높다

---

## 8.5 text prior branch
Discussion EDA상 텍스트 편향이 크다.  
따라서 **text-only prior**는 반드시 두는 게 좋다.

구성:
- 입력: question + [SEP] + a/b/c/d
- 모델: char/word TF-IDF + linear / LightGBM / small transformer
- 출력: option prior score

결합:
- material/object/recycle/location에서는 가중치 높게
- count에서는 가중치 낮게

---

## 9. CV / 검증 설계

리더보드와 내부 CV가 다르게 느껴지는 이유는 질문 템플릿 반복 때문이다.

### 권장
- 단순 StratifiedKFold만 쓰지 말고
- **normalized question template group split**을 같이 실험

예:
- "사진에 보이는 재활용 가능한 플라스틱 병은 몇 개입니까?"
- "사진 속 재활용품 중 플라스틱 병은 몇 개 있나요?"
- 둘 다 `count_bottle` 계열 템플릿으로 묶기

왜 필요하나:
- 텍스트 편향이 큰 문제에서 일반 KFold는 지나치게 낙관적일 수 있음
- template-aware CV가 실제 test 일반화와 더 가까울 가능성이 높음

---

## 10. 최종 앙상블 규칙

### 10.1 q_type별 기본 결합
- material / object / recycle:
  - `0.45 * material_object_expert`
  - `0.30 * shared`
  - `0.20 * text_prior`
  - `0.05 * OCR/heuristic`

- count:
  - `0.45 * count_expert`
  - `0.25 * shared`
  - `0.20 * detector/count heuristic`
  - `0.10 * text_prior`

- color:
  - `0.45 * shared`
  - `0.25 * material_object_expert`
  - `0.20 * text_prior`
  - `0.10 * crop recheck`

- location / brand / other:
  - `0.35 * shared`
  - `0.25 * text_prior`
  - `0.25 * OCR/location heuristic`
  - `0.15 * fallback model`

---

### 10.2 불확실 샘플 2차 추론
조건:
- top1-top2 margin이 작음
- q_type이 location / brand / other / count
- OCR 신호가 강한데 VLM과 충돌

이때만 추가 수행:
- high-res 재추론
- crop view
- 다른 backbone(Qwen3-VL ↔ Qwen3.5) 재판정
- OCR score 반영

---

## 11. 4일 실전 플랜

## Day 1
- q_type router 완성
- answer-only masking 적용
- logprob reranking 적용
- aspect-ratio preserving pipeline로 교체
- train only shared baseline 확보

## Day 2
- AI Hub 생활 폐기물 synthetic 생성
- material/object expert 학습
- text prior 모델 학습
- option shuffle 적용

## Day 3
- TACO / bbox 기반 count synthetic 생성
- dev soft pseudo label 생성
- count expert 학습
- OCR 보조 파이프라인 연결

## Day 4
- Qwen3.5-9B vs Qwen3-VL-8B 비교
- q_type별 ensemble weight 탐색
- option permutation averaging
- uncertain-only heavy inference
- 제출본 3~5개 관리

---

## 12. 우선순위 체크리스트

### 무조건 할 것
- [ ] 384 stretch 제거
- [ ] answer-only masking
- [ ] logprob reranking
- [ ] option shuffle augmentation
- [ ] q_type router
- [ ] text prior branch

### 점수 상승폭이 큰 것
- [ ] AI Hub material/object synthetic
- [ ] count expert + dev soft label
- [ ] TACO count synthetic
- [ ] q_type별 앙상블

### 시간 남으면 할 것
- [ ] OCR expert 개선
- [ ] Qwen3.5-9B 실험
- [ ] uncertain-only 2차 추론
- [ ] template-aware CV

---

## 13. 최종 추천

시간이 매우 부족하면:
- **Qwen3-VL-8B + aspect ratio 유지 + reranking + text prior + material/object synthetic**  
이 조합이 가장 안전하다.

정말 1등 ceiling을 노리면:
- **Qwen3.5-9B shared**
- **Qwen3-VL-8B fallback**
- **material/object expert**
- **count expert**
- **AI Hub synthetic**
- **dev soft pseudo label**
- **q_type별 ensemble**
로 가는 게 가장 강하다.

핵심은 한 문장으로 요약된다.

> 이 대회는 “범용 VQA 한 방”이 아니라,  
> **material/object 중심 외부데이터 강화 + count 분업 + 텍스트 편향 활용 + 고해상도 추론**으로 이겨야 한다.
