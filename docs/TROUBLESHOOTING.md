# 🔧 Korean Bllossom AICA-5B 문제 해결 가이드

이 문서는 Korean Bllossom AICA-5B 양자화 프로젝트 사용 중 발생할 수 있는 문제들과 해결 방법을 제공합니다.

## 📋 목차

- [설치 관련 문제](#설치-관련-문제)
- [모델 로딩 문제](#모델-로딩-문제)
- [메모리 관련 문제](#메모리-관련-문제)
- [성능 관련 문제](#성능-관련-문제)
- [네트워크 관련 문제](#네트워크-관련-문제)
- [하드웨어 관련 문제](#하드웨어-관련-문제)
- [자주 묻는 질문](#자주-묻는-질문)
- [진단 도구](#진단-도구)

## 🛠️ 설치 관련 문제

### 1. CUDA 관련 오류

#### 문제: `RuntimeError: CUDA runtime error`

**증상**:
```
RuntimeError: CUDA runtime error (2) : out of memory
```

**원인**: 
- CUDA 드라이버와 PyTorch 버전 불일치
- 구버전 CUDA 사용

**해결책**:

```bash
# 1. CUDA 버전 확인
nvidia-smi
nvcc --version

# 2. PyTorch 재설치 (CUDA 12.1 기준)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. CUDA 11.8인 경우
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. 설치 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 문제: `NVIDIA-SMI has failed`

**증상**:
```bash
$ nvidia-smi
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

**해결책**:

```bash
# 1. 드라이버 상태 확인
lsmod | grep nvidia

# 2. 자동 드라이버 설치
sudo ubuntu-drivers autoinstall

# 3. 수동 드라이버 설치
sudo apt update
sudo apt install nvidia-driver-525  # 또는 최신 버전

# 4. 시스템 재부팅
sudo reboot

# 5. 확인
nvidia-smi
```

### 2. Python 패키지 설치 오류

#### 문제: `bitsandbytes` 설치 실패

**증상**:
```
ERROR: Failed building wheel for bitsandbytes
```

**해결책**:

```bash
# 방법 1: 사전 컴파일된 버전 설치
pip install bitsandbytes --no-cache-dir

# 방법 2: CUDA 경로 설정 후 설치
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip install bitsandbytes --no-binary bitsandbytes

# 방법 3: conda 사용
conda install -c conda-forge bitsandbytes

# 방법 4: 특정 버전 설치
pip install bitsandbytes==0.41.0
```

#### 문제: `transformers` 버전 충돌

**증상**:
```
AttributeError: module 'transformers' has no attribute 'MllamaForConditionalGeneration'
```

**해결책**:

```bash
# 1. 최신 버전으로 업그레이드
pip install transformers --upgrade

# 2. 특정 버전 강제 설치
pip install transformers==4.40.0 --force-reinstall

# 3. 개발 버전 설치 (최신 기능 필요시)
pip install git+https://github.com/huggingface/transformers.git

# 4. 설치 확인
python -c "from transformers import MllamaForConditionalGeneration; print('OK')"
```

### 3. 가상환경 문제

#### 문제: 가상환경에서 패키지를 찾을 수 없음

**해결책**:

```bash
# 1. 가상환경 재생성
rm -rf bllossom_env
python3 -m venv bllossom_env
source bllossom_env/bin/activate

# 2. pip 업그레이드
pip install --upgrade pip setuptools wheel

# 3. 패키지 재설치
pip install -r requirements.txt

# 4. 환경 확인
which python
which pip
```

## 🤖 모델 로딩 문제

### 1. 모델 다운로드 실패

#### 문제: `ConnectionError` 또는 다운로드 중단

**해결책**:

```bash
# 1. 네트워크 확인
ping huggingface.co

# 2. 캐시 디렉토리 정리
rm -rf ~/.cache/huggingface/
rm -rf ./model_cache/

# 3. 수동 다운로드
python -c "
from transformers import MllamaProcessor
processor = MllamaProcessor.from_pretrained(
    'Bllossom/llama-3.2-Korean-Bllossom-AICA-5B',
    cache_dir='./model_cache',
    resume_download=True
)
print('다운로드 완료!')
"

# 4. 프록시 설정 (회사 네트워크인 경우)
export https_proxy=http://proxy.company.com:8080
export http_proxy=http://proxy.company.com:8080
```

#### 문제: `OSError: Can't load tokenizer`

**해결책**:

```bash
# 1. 토크나이저만 별도 다운로드
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'Bllossom/llama-3.2-Korean-Bllossom-AICA-5B',
    trust_remote_code=True
)
"

# 2. 캐시 권한 확인
sudo chown -R $USER ~/.cache/huggingface/
chmod -R 755 ~/.cache/huggingface/

# 3. 환경 변수 설정
export HF_HOME=./model_cache
export TRANSFORMERS_CACHE=./model_cache
```

### 2. 양자화 로딩 오류

#### 문제: `ImportError: bitsandbytes`

**증상**:
```
ImportError: bitsandbytes is not installed. Please install it with `pip install bitsandbytes`.
```

**해결책**:

```python
# config.yaml에서 양자화 비활성화
quantization:
  load_in_4bit: false

# 또는 Python 코드에서
config.quantization.load_in_4bit = False
```

```bash
# bitsandbytes 재설치
pip uninstall bitsandbytes
pip install bitsandbytes --no-cache-dir
```

## 💾 메모리 관련 문제

### 1. VRAM 부족 (OOM) 오류

#### 문제: `torch.cuda.OutOfMemoryError`

**증상**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**해결책 (우선순위대로)**:

```bash
# 1. 다른 GPU 프로세스 종료
nvidia-smi
sudo kill -9 <PID>

# 2. 시스템 재부팅
sudo reboot
```

```python
# 3. 메모리 설정 조정
config.generation.max_tokens = 128  # 기본값 256에서 줄임
config.hardware.max_memory_usage = 0.7  # 70%만 사용

# 4. 양자화 강화
config.quantization.load_in_4bit = True
config.quantization.bnb_4bit_compute_dtype = "float16"

# 5. 캐시 비활성화
use_cache = False

# 6. 배치 크기 줄이기
# 한 번에 하나씩만 처리
```

```bash
# 7. 환경 변수 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### 문제: 시스템 RAM 부족

**해결책**:

```bash
# 1. 스왑 메모리 생성
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 2. 영구 설정
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 3. 메모리 사용량 확인
free -h
```

### 2. 메모리 누수

#### 문제: 메모리 사용량이 계속 증가

**해결책**:

```python
# 1. 명시적 메모리 정리
import gc
import torch

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# 2. 작업 후 항상 메모리 정리
result = generator.generate("prompt")
clear_memory()

#