#!/bin/bash

# Korean Bllossom AICA-5B 양자화 프로젝트 설치 스크립트
# Ubuntu 22.04 기준

set -e  # 오류 발생 시 스크립트 중단

echo "🌸 Korean Bllossom AICA-5B 양자화 프로젝트 설치"
echo "================================================"
echo "시스템: Ubuntu 22.04"
echo "타겟 GPU: RTX 4060 8GB"
echo "================================================"

# 색깔 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로깅 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 시스템 요구사항 확인
check_system() {
    log_info "시스템 요구사항 확인 중..."
    
    # Ubuntu 버전 확인
    if ! grep -q "22.04" /etc/os-release; then
        log_warning "Ubuntu 22.04가 아닙니다. 호환성 문제가 있을 수 있습니다."
    fi
    
    # Python 버전 확인
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [ "$(echo "$PYTHON_VERSION < 3.8" | bc)" -eq 1 ]; then
        log_error "Python 3.8 이상이 필요합니다. 현재: $PYTHON_VERSION"
        exit 1
    fi
    log_success "Python 버전 확인: $PYTHON_VERSION"
    
    # 메모리 확인
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_RAM" -lt 16 ]; then
        log_warning "RAM이 ${TOTAL_RAM}GB입니다. 16GB 이상 권장합니다."
    else
        log_success "RAM 확인: ${TOTAL_RAM}GB"
    fi
    
    # 디스크 공간 확인
    FREE_SPACE=$(df . | tail -1 | awk '{print $4}')
    FREE_SPACE_GB=$((FREE_SPACE / 1024 / 1024))
    if [ "$FREE_SPACE_GB" -lt 50 ]; then
        log_warning "여유 디스크 공간이 ${FREE_SPACE_GB}GB입니다. 50GB 이상 권장합니다."
    else
        log_success "디스크 공간 확인: ${FREE_SPACE_GB}GB"
    fi
}

# 시스템 패키지 업데이트
update_system() {
    log_info "시스템 패키지 업데이트 중..."
    sudo apt update -qq
    sudo apt upgrade -y -qq
    log_success "시스템 업데이트 완료"
}

# 필수 패키지 설치
install_system_packages() {
    log_info "시스템 패키지 설치 중..."
    
    PACKAGES=(
        python3-pip
        python3-venv
        python3-dev
        build-essential
        curl
        wget
        git
        htop
        nvtop
        tree
        unzip
        software-properties-common
        apt-transport-https
        ca-certificates
        gnupg
        lsb-release
    )
    
    for package in "${PACKAGES[@]}"; do
        if ! dpkg -l | grep -q "^ii  $package "; then
            log_info "설치 중: $package"
            sudo apt install -y "$package" -qq
        else
            log_info "이미 설치됨: $package"
        fi
    done
    
    log_success "시스템 패키지 설치 완료"
}

# NVIDIA 드라이버 확인 및 설치
setup_nvidia() {
    log_info "NVIDIA 설정 확인 중..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "NVIDIA 드라이버가 설치되지 않았습니다!"
        echo "다음 중 하나를 선택하세요:"
        echo "1. 자동 설치 (권장)"
        echo "2. 수동 설치 후 재시작"
        echo "3. 건너뛰기 (CPU 모드)"
        
        read -p "선택 (1-3): " choice
        case $choice in
            1)
                log_info "NVIDIA 드라이버 자동 설치 중..."
                sudo ubuntu-drivers autoinstall
                log_warning "설치 완료 후 시스템을 재부팅해야 합니다."
                echo "재부팅 후 다시 이 스크립트를 실행하세요: ./setup.sh --resume"
                exit 0
                ;;
            2)
                log_info "다음 명령어로 수동 설치하세요:"
                echo "sudo ubuntu-drivers devices"
                echo "sudo ubuntu-drivers autoinstall"
                echo "sudo reboot"
                exit 0
                ;;
            3)
                log_warning "CPU 모드로 계속합니다. 성능이 매우 느릴 수 있습니다."
                ;;
        esac
    else
        log_success "NVIDIA 드라이버 확인됨"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        
        # VRAM 확인
        VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        VRAM_GB=$((VRAM / 1024))
        
        if [ "$VRAM_GB" -lt 6 ]; then
            log_warning "VRAM이 ${VRAM_GB}GB입니다. 6GB 이상 권장합니다."
        else
            log_success "VRAM 확인: ${VRAM_GB}GB"
        fi
    fi
}

# Python 가상환경 설정
setup_python_env() {
    log_info "Python 가상환경 설정 중..."
    
    ENV_NAME="bllossom_env"
    
    # 기존 환경 제거 (선택사항)
    if [ -d "$ENV_NAME" ]; then
        read -p "기존 가상환경을 재생성하시겠습니까? (y/n): " recreate
        if [[ $recreate =~ ^[Yy]$ ]]; then
            rm -rf "$ENV_NAME"
            log_info "기존 가상환경 제거됨"
        fi
    fi
    
    # 가상환경 생성
    if [ ! -d "$ENV_NAME" ]; then
        log_info "가상환경 생성 중..."
        python3 -m venv "$ENV_NAME"
        log_success "가상환경 생성 완료: $ENV_NAME"
    fi
    
    # 가상환경 활성화
    source "$ENV_NAME/bin/activate"
    log_success "가상환경 활성화됨"
    
    # pip 업그레이드
    log_info "pip 업그레이드 중..."
    pip install --upgrade pip setuptools wheel
    log_success "pip 업그레이드 완료"
}

# PyTorch 설치
install_pytorch() {
    log_info "PyTorch 설치 중..."
    
    # CUDA 버전 확인
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
        log_info "감지된 CUDA 버전: $CUDA_VERSION"
        
        # CUDA 버전에 따른 PyTorch 설치
        case $CUDA_VERSION in
            "12.1"|"12.2"|"12.3"|"12.4")
                log_info "CUDA 12.x용 PyTorch 설치..."
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
                ;;
            "11.8"|"11.7")
                log_info "CUDA 11.x용 PyTorch 설치..."
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
                ;;
            *)
                log_warning "지원되지 않는 CUDA 버전입니다. CPU 버전을 설치합니다."
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
                ;;
        esac
    else
        log_info "CPU용 PyTorch 설치..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    log_success "PyTorch 설치 완료"
}

# 프로젝트 의존성 설치
install_dependencies() {
    log_info "프로젝트 의존성 설치 중..."
    
    if [ ! -f "requirements.txt" ]; then
        log_error "requirements.txt 파일을 찾을 수 없습니다!"
        exit 1
    fi
    
    # 핵심 의존성만 먼저 설치
    log_info "핵심 의존성 설치 중..."
    pip install transformers accelerate bitsandbytes
    
    # 나머지 의존성 설치
    log_info "추가 의존성 설치 중..."
    pip install -r requirements.txt
    
    log_success "의존성 설치 완료"
}

# 설치 확인
verify_installation() {
    log_info "설치 확인 중..."
    
    python3 << 'EOF'
import sys
import torch
import transformers
import accelerate
import bitsandbytes

print("✅ 설치 확인 결과:")
print(f"   Python: {sys.version}")
print(f"   PyTorch: {torch.__version__}")
print(f"   Transformers: {transformers.__version__}")
print(f"   Accelerate: {accelerate.__version__}")
print(f"   BitsAndBytes: {bitsandbytes.__version__}")

print(f"\n🎮 GPU 정보:")
print(f"   CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA 버전: {torch.version.cuda}")
    print(f"   GPU 개수: {torch.cuda.device_count()}")
    print(f"   GPU 이름: {torch.cuda.get_device_name()}")
    
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"   VRAM: {total_memory / 1024**3:.1f}GB")
    
    # 간단한 CUDA 테스트
    try:
        x = torch.tensor([1.0]).cuda()
        print(f"   CUDA 테스트: 성공")
    except Exception as e:
        print(f"   CUDA 테스트: 실패 - {e}")
else:
    print("   CPU 모드로 실행됩니다.")
EOF
    
    if [ $? -eq 0 ]; then
        log_success "설치 확인 완료"
    else
        log_error "설치 확인 실패"
        return 1
    fi
}

# 프로젝트 파일 권한 설정
setup_permissions() {
    log_info "파일 권한 설정 중..."
    
    # Python 파일 실행 권한
    find . -name "*.py" -exec chmod +x {} \;
    
    # 스크립트 파일 실행 권한
    find . -name "*.sh" -exec chmod +x {} \;
    
    log_success "권한 설정 완료"
}

# 프로젝트 디렉토리 생성
create_directories() {
    log_info "프로젝트 디렉토리 생성 중..."
    
    DIRECTORIES=(
        "model_cache"
        "logs"
        "outputs"
        "data"
        "experiments"
        "checkpoints"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "디렉토리 생성: $dir"
        fi
    done
    
    log_success "디렉토리 생성 완료"
}

# 설정 파일 생성
create_config() {
    log_info "기본 설정 파일 생성 중..."
    
    if [ ! -f "config.yaml" ]; then
        python3 -c "
from config import config
config.save_to_yaml('config.yaml')
print('✅ config.yaml 생성 완료')
"
    else
        log_info "config.yaml이 이미 존재합니다."
    fi
}

# 메인 설치 함수
main_install() {
    echo "설치를 시작합니다..."
    echo "예상 소요 시간: 10-20분"
    echo ""
    
    # 설치 단계 실행
    check_system
    update_system
    install_system_packages
    setup_nvidia
    setup_python_env
    install_pytorch
    install_dependencies
    setup_permissions
    create_directories
    verify_installation
    
    log_success "🎉 설치 완료!"
    
    echo ""
    echo "================================================"
    echo "🚀 사용법:"
    echo "1. 가상환경 활성화: source bllossom_env/bin/activate"
    echo "2. 시스템 확인: python main.py --check"
    echo "3. 데모 실행: python main.py --demo"
    echo "4. 채팅 모드: python main.py --chat"
    echo "5. CLI 도움말: python cli_interface.py --help"
    echo ""
    echo "📚 추가 정보:"
    echo "- 설정 파일: config.yaml"
    echo "- 로그 디렉토리: logs/"
    echo "- 모델 캐시: model_cache/"
    echo "================================================"
}

# 재시작 후 설치 함수
resume_install() {
    log_info "설치 재개 중..."
    setup_python_env
    install_pytorch
    install_dependencies
    verify_installation
    log_success "설치 재개 완료!"
}

# 인수 처리
case "${1:-}" in
    --resume)
        resume_install
        ;;
    --check)
        check_system
        setup_nvidia
        ;;
    --clean)
        log_info "설치 파일 정리 중..."
        rm -rf bllossom_env model_cache logs outputs
        log_success "정리 완료"
        ;;
    *)
        main_install
        ;;
esac
