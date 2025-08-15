# 📦 Korean Bllossom AICA-5B 설치 가이드

이 문서는 Korean Bllossom AICA-5B 양자화 프로젝트의 상세한 설치 가이드입니다.

## 📋 시스템 요구사항

### 최소 요구사항

| 구성 요소 | 최소 사양 | 권장 사양 |
|-----------|-----------|-----------|
| **OS** | Ubuntu 20.04+ | Ubuntu 22.04 LTS |
| **Python** | 3.8+ | 3.10+ |
| **GPU** | 6GB VRAM | 8GB+ VRAM |
| **RAM** | 16GB | 32GB+ |
| **저장공간** | 30GB | 50GB+ |
| **CUDA** | 11.7+ | 12.1+ |

### 지원 GPU 목록

#### ✅ 완전 지원
- NVIDIA RTX 4060 Ti (16GB) - **최적**
- NVIDIA RTX 4070 (12GB) - **최적**
- NVIDIA RTX 4060 (8GB) - **최적화 타겟**
- NVIDIA RTX 3070 Ti (8GB)
- NVIDIA RTX 3080 (10GB)

#### ⚠️ 제한적 지원
- NVIDIA RTX 3060 (12GB) - 느린 속도
- NVIDIA RTX 3060 Ti (8GB) - 메모리 부족 가능
- NVIDIA GTX 1080 Ti (11GB) - 양자화 필수

#### ❌ 지원 불가
- 6GB 미만 VRAM GPU
- AMD GPU (현재 미지원)
- Intel GPU (현재 미지원)

## 🛠️ 설치 방법

### 방법 1: 자동 설치 (권장)

가장 간단하고 안전한 설치 방법입니다.

```bash
# 1. 프로젝트 다운로드
git clone <repository-url>
cd korean-bllossom-quantized

# 2. 자동 설치 실행
chmod +x setup.sh
./setup.sh

# 3. 설치 완료 후 확인
source bllossom_env/bin/activate
python main.py --check
```

### 방법 2: 수동 설치

세부적인 제어가 필요한 경우에 사용합니다.

#### 2.1 시스템 패키지 설치

```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 패키지 설치
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    curl \
    wget \
    git \
    htop \
    nvtop
```

#### 2.2 NVIDIA 드라이버 설치

```bash
# 현재 드라이버 확인
nvidia-smi

# 드라이버가 없는 경우 설치
sudo ubuntu-drivers autoinstall
sudo reboot
```

#### 2.3 Python 가상환경 생성

```bash
# 가상환경 생성
python3 -m venv bllossom_env

# 가상환경 활성화
source bllossom_env/bin/activate

# pip 업그레이드
pip install --upgrade pip setuptools wheel
```

#### 2.4 PyTorch 설치

CUDA 버전에 따라 적절한 PyTorch를 설치합니다.

```bash
# CUDA 버전 확인
nvidia-smi | grep "CUDA Version"

# CUDA 12.1+ 환경
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 환경
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU 전용 (GPU 없는 경우)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 2.5 프로젝트 의존성 설치

```bash
# 핵심 의존성 먼저 설치
pip install transformers accelerate bitsandbytes

# 나머지 의존성 설치
pip install -r requirements.txt
```

#### 2.6 설치 확인

```bash
python3 -c "
import torch
import transformers
import bitsandbytes
print('✅ 모든 패키지 설치 완료')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
"
```

### 방법 3: Docker 설치 (고급)

Docker를 사용한 컨테이너 환경에서 실행하는 방법입니다.

#### 3.1 Docker 준비

```bash
# Docker 설치 (Ubuntu)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# NVIDIA Container Toolkit 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### 3.2 Docker 이미지 빌드

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 프로젝트 파일 복사
COPY . .

# Python 의존성 설치
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install -r requirements.txt

# 실행 명령
CMD ["python3", "main.py", "--demo"]
```

```bash
# 이미지 빌드
docker build -t korean-bllossom .

# 컨테이너 실행
docker run --gpus all -it korean-bllossom
```

## 🔧 설치 문제 해결

### 일반적인 설치 오류

#### 1. CUDA 버전 불일치

**증상**: `RuntimeError: CUDA runtime error`

**해결책**:
```bash
# CUDA 버전 확인
nvcc --version
nvidia-smi

# PyTorch 재설치 (적절한 CUDA 버전으로)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 2. bitsandbytes 설치 실패

**증상**: `ERROR: Failed building wheel for bitsandbytes`

**해결책**:
```bash
# 사전 컴파일된 버전 설치
pip install bitsandbytes --no-cache-dir

# 또는 소스에서 컴파일
export CUDA_HOME=/usr/local/cuda
pip install bitsandbytes --no-binary bitsandbytes
```

#### 3. transformers 버전 충돌

**증상**: `AttributeError: module 'transformers' has no attribute`

**해결책**:
```bash
# 최신 버전으로 업그레이드
pip install transformers --upgrade

# 또는 특정 버전 설치
pip install transformers==4.40.0
```

#### 4. 메모리 부족 오류

**증상**: 설치 중 시스템이 멈춤

**해결책**:
```bash
# 스왑 메모리 증가
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# pip 캐시 정리
pip cache purge
```

### 특정 환경별 해결책

#### Ubuntu 20.04

```bash
# Python 3.10 설치 (필요한 경우)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# python3.10으로 가상환경 생성
python3.10 -m venv bllossom_env
```

#### WSL2 (Windows)

```bash
# WSL2에서 CUDA 설정
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Windows 방화벽 설정 (웹 인터페이스 사용 시)
# Windows 설정 > 네트워크 및 인터넷 > Windows Defender 방화벽
# 포트 7860, 8000 허용 추가
```

#### macOS (Apple Silicon)

현재 공식적으로 지원되지 않지만, CPU 모드로 실행 가능합니다.

```bash
# Homebrew를 통한 Python 설치
brew install python@3.10

# PyTorch CPU 버전 설치
pip install torch torchvision torchaudio

# bitsandbytes 건너뛰기 (Apple Silicon 미지원)
pip install -r requirements.txt --ignore-installed bitsandbytes
```

## ⚙️ 고급 설정

### 성능 최적화 설정

#### 1. CUDA 최적화

```bash
# ~/.bashrc에 추가
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
```

#### 2. 메모리 최적화

```python
# config.yaml 수정
hardware:
  max_memory_usage: 0.85  # VRAM의 85%만 사용
  use_cpu_offload: true   # 필요시 CPU 오프로드 사용
  
generation:
  max_tokens: 200         # 토큰 수 제한
  use_cache: false        # 캐시 비활성화로 메모리 절약
```

#### 3. 네트워크 최적화

```bash
# Hugging Face 캐시 설정
export HF_HOME=./model_cache
export HF_HUB_CACHE=./model_cache
export TRANSFORMERS_CACHE=./model_cache

# 프록시 설정 (필요한 경우)
export https_proxy=http://proxy.company.com:8080
export http_proxy=http://proxy.company.com:8080
```

### 개발 환경 설정

#### 개발 의존성 설치

```bash
# 개발 도구 설치
pip install \
    black \
    flake8 \
    pytest \
    jupyter \
    ipywidgets \
    matplotlib \
    seaborn
```

#### IDE 설정

**VS Code 확장 프로그램**:
- Python
- Pylance
- Jupyter
- GitLens
- Thunder Client (API 테스트)

**PyCharm 설정**:
- 인터프리터: `./bllossom_env/bin/python`
- 소스 루트: 프로젝트 루트 디렉토리
- 실행 구성: `main.py` 스크립트

## 🧪 설치 확인 및 테스트

### 기본 테스트

```bash
# 1. 시스템 요구사항 확인
python main.py --check

# 2. 간단한 데모 실행
python main.py --demo

# 3. 모델 로딩 테스트
python -c "
from model_manager import get_model_manager
from config import Config
manager = get_model_manager(Config())
print('✅ 모델 매니저 초기화 성공')
"
```

### 성능 벤치마크

```bash
# 성능 측정 스크립트 실행
python scripts/benchmark.py --gpu-test --memory-test

# 예상 결과:
# ✅ GPU 감지: RTX 4060 8GB
# ✅ VRAM 사용량: 5.2GB / 8.0GB
# ✅ 생성 속도: 22.3 토큰/초
# ✅ 메모리 효율성: 85%
```

### 문제 진단

```bash
# 상세 진단 스크립트
python scripts/diagnose.py

# 로그 파일 확인
tail -f logs/install.log
tail -f logs/model.log
```

## 📚 추가 리소스

### 공식 문서
- [Transformers 문서](https://huggingface.co/docs/transformers)
- [BitsAndBytes 문서](https://github.com/TimDettmers/bitsandbytes)
- [PyTorch 설치 가이드](https://pytorch.org/get-started/)

### 커뮤니티 지원
- [GitHub Issues](링크)
- [Discussion Forum](링크)
- [Discord 채널](링크)

### 비디오 튜토리얼
- [설치 가이드 영상](링크)
- [사용법 데모](링크)
- [문제 해결](링크)

---

설치 과정에서 문제가 발생하면 GitHub Issues에 다음 정보와 함께 문의해주세요:

1. **시스템 정보**: OS, Python 버전, GPU 모델
2. **오류 메시지**: 전체 오류 로그
3. **설치 단계**: 어느 단계에서 실패했는지
4. **환경 변수**: CUDA_HOME, PATH 등

**🔧 빠른 도움이 필요하다면 자동 진단 도구를 실행하세요:**
```bash
python scripts/diagnose.py --create-report
```
