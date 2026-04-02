# ssafy_ai_challenge

# 🛠️ 개발 환경 세팅 가이드 (uv)

Python 패키지 매니저 **uv**를 사용하여 PyTorch, HuggingFace 등 딥러닝 환경을 구축하는 가이드입니다.  
CUDA 지원 여부와 운영체제에 상관없이 동일한 프로젝트를 사용할 수 있습니다.

---

## 📋 환경별 요약

| 환경 | OS | CUDA | 설치 명령어 |
|---|---|---|---|
| SSAFY 강의장 | Windows | ✅ 지원 (RTX 등) | `uv sync --extra cu130` |
| 개인 PC | Windows | ❌ 미지원 | `uv sync --extra cpu` |
| 개인 PC | macOS | ❌ (MPS) | `uv sync --extra cpu` |

---

## ✅ Step 0. 사전 확인 (Windows + CUDA 환경만)

SSAFY 강의장 등 CUDA를 사용하는 경우, 아래 명령어로 드라이버 버전을 먼저 확인합니다.

```powershell
nvidia-smi
```

출력 우측 상단의 `CUDA Version` 숫자가 사용 가능한 최대 버전입니다.

| CUDA Version | 사용할 백엔드 |
|---|---|
| 13.x | `cu130` |
| 12.8 | `cu128` |
| 12.6 | `cu126` |

> 이 프로젝트의 `pyproject.toml`은 **cu130** 기준으로 설정되어 있습니다.  
> 다른 버전이 필요한 경우 `pyproject.toml`의 `cu130` 부분을 수정하세요.

---

## ✅ Step 1. uv 설치

### Windows

PowerShell을 열고 아래 명령어를 실행합니다.

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 설치 확인

> ⚠️ 설치 후 **반드시 터미널을 새로 열고** 아래 명령어로 확인하세요.

```bash
uv --version
# uv 0.x.x 출력되면 성공
```

---

## ✅ Step 2. Python 설치

```bash
uv python install 3.12
```

> Python 3.13은 PyTorch와 호환성 문제가 있으므로 **3.12를 권장**합니다.

---

## ✅ Step 3. 프로젝트 클론 및 이동

```bash
git clone <repository-url>
cd <project-folder>
```

---

## ✅ Step 4. 환경별 패키지 설치

**내 환경에 맞는 명령어 하나만 실행하면 됩니다.**

### 🖥️ SSAFY 강의장 (Windows + CUDA)

```powershell
uv sync --extra cu130
```

### 💻 개인 PC — Windows (CUDA 미지원)

```powershell
uv sync --extra cpu
```

### 🍎 개인 PC — macOS

```bash
uv sync --extra cpu
```

---

## ✅ Step 5. 가상환경 활성화

`uv sync` 실행 시 `.venv` 폴더가 자동으로 생성됩니다.  
아래 명령어로 활성화하세요.

### Windows

```powershell
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

---

## ✅ Step 6. 설치 확인

```python
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

**SSAFY 강의장 (CUDA)** 정상 출력 예시:
```
PyTorch: 2.x.x+cu130
CUDA available: True
```

**개인 PC (CPU)** 정상 출력 예시:
```
PyTorch: 2.x.x+cpu
CUDA available: False
```

---

## 📦 패키지 추가하기

새로운 패키지가 필요하면 아래 명령어를 사용하세요.  
`pyproject.toml`이 자동으로 업데이트되고, `git push` 후 팀원들도 `uv sync`만 하면 반영됩니다.

```bash
# 패키지 추가
uv add <패키지명>

# 예시
uv add diffusers
uv add opencv-python
```

### ⚠️ CUDA 환경에서 패키지 추가 시 주의사항

`uv add`는 extra 지정 없이 의존성을 re-resolve하기 때문에, **실행 후 torch가 CPU 버전으로 다운그레이드될 수 있습니다.**

반드시 `uv add` 후에 아래 명령어로 마무리하세요.

```powershell
# SSAFY 강의장 (CUDA 환경)
uv add <패키지명> && uv sync --extra cu130

# 개인 PC (CPU 환경)
uv add <패키지명> && uv sync --extra cpu
```

torch가 다운그레이드된 경우 아래와 같이 롤백할 수 있습니다.

```powershell
uv remove <추가한 패키지명>
uv sync --extra cu130
```

그 후 다시 `uv add <패키지명> && uv sync --extra cu130` 순서로 진행하세요.

> ⚠️ `torch`, `torchvision`, `torchaudio`는 CUDA 인덱스 설정이 별도로 필요하므로  
> `uv add`로 추가하지 말고, `pyproject.toml`을 직접 수정하거나 팀장에게 요청하세요.

---

## 🔧 자주 쓰는 명령어

```bash
# 패키지 목록 확인
uv pip list

# 의존성 트리 확인
uv tree

# 패키지 제거
uv remove <패키지명>

# uv 자체 업데이트
uv self update

# 환경 재설치 (문제 발생 시)
uv sync --reinstall
```

---

## 📁 프로젝트 구조

```
project/
├── .venv/             # 가상환경 (git 제외)
├── pyproject.toml     # 의존성 설정 파일 (git 포함)
├── uv.lock            # 버전 고정 파일 (git 포함)
└── README.md
```

> `.venv/`는 `.gitignore`에 추가되어 있어야 합니다.  
> `pyproject.toml`과 `uv.lock`은 반드시 git에 포함하세요.

---

## ❓ 문제 해결

**`uv` 명령어를 찾을 수 없다는 오류가 뜨는 경우**
→ 터미널을 새로 열고 다시 시도하세요.

**패키지 설치 중 오류가 발생하는 경우**
```bash
# 캐시 초기화 후 재시도
uv cache prune
uv sync --extra cu130   # 또는 --extra cpu
```

**CUDA available: False 가 뜨는 경우 (SSAFY 강의장)**
→ `uv sync --extra cpu`가 아닌 `uv sync --extra cu130`으로 설치했는지 확인하세요.  
→ NVIDIA 드라이버가 정상적으로 설치되어 있는지 `nvidia-smi`로 확인하세요.