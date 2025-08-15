# config.py
# 프로젝트 설정 관리

import os
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """모델 설정"""
    name: str = "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    low_cpu_mem_usage: bool = True
    device_map: str = "auto"

@dataclass
class QuantizationConfig:
    """양자화 설정"""
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

@dataclass
class GenerationConfig:
    """텍스트 생성 설정"""
    max_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

@dataclass
class HardwareConfig:
    """하드웨어 설정"""
    target_gpu: str = "RTX 4060"
    target_vram_gb: int = 8
    use_cpu_offload: bool = False
    max_memory_usage: float = 0.9  # VRAM의 90%까지 사용

@dataclass
class PathConfig:
    """경로 설정"""
    cache_dir: str = "./model_cache"
    log_dir: str = "./logs"
    output_dir: str = "./outputs"
    config_file: str = "./config.yaml"

class Config:
    """전체 설정 관리 클래스"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.model = ModelConfig()
        self.quantization = QuantizationConfig()
        self.generation = GenerationConfig()
        self.hardware = HardwareConfig()
        self.paths = PathConfig()
        
        if config_file and os.path.exists(config_file):
            self.load_from_yaml(config_file)
        
        # 디렉토리 생성
        self._create_directories()
    
    def load_from_yaml(self, config_file: str):
        """YAML 파일에서 설정 로드"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            # 각 섹션별로 설정 업데이트
            if 'model' in config_dict:
                for key, value in config_dict['model'].items():
                    if hasattr(self.model, key):
                        setattr(self.model, key, value)
            
            if 'quantization' in config_dict:
                for key, value in config_dict['quantization'].items():
                    if hasattr(self.quantization, key):
                        setattr(self.quantization, key, value)
            
            if 'generation' in config_dict:
                for key, value in config_dict['generation'].items():
                    if hasattr(self.generation, key):
                        setattr(self.generation, key, value)
            
            if 'hardware' in config_dict:
                for key, value in config_dict['hardware'].items():
                    if hasattr(self.hardware, key):
                        setattr(self.hardware, key, value)
            
            if 'paths' in config_dict:
                for key, value in config_dict['paths'].items():
                    if hasattr(self.paths, key):
                        setattr(self.paths, key, value)
                        
        except Exception as e:
            print(f"⚠️ 설정 파일 로드 실패: {e}")
            print("기본 설정을 사용합니다.")
    
    def save_to_yaml(self, config_file: str = None):
        """현재 설정을 YAML 파일로 저장"""
        if config_file is None:
            config_file = self.paths.config_file
        
        config_dict = {
            'model': {
                'name': self.model.name,
                'trust_remote_code': self.model.trust_remote_code,
                'torch_dtype': self.model.torch_dtype,
                'low_cpu_mem_usage': self.model.low_cpu_mem_usage,
                'device_map': self.model.device_map
            },
            'quantization': {
                'load_in_4bit': self.quantization.load_in_4bit,
                'bnb_4bit_quant_type': self.quantization.bnb_4bit_quant_type,
                'bnb_4bit_compute_dtype': self.quantization.bnb_4bit_compute_dtype,
                'bnb_4bit_use_double_quant': self.quantization.bnb_4bit_use_double_quant
            },
            'generation': {
                'max_tokens': self.generation.max_tokens,
                'temperature': self.generation.temperature,
                'do_sample': self.generation.do_sample,
                'top_p': self.generation.top_p,
                'top_k': self.generation.top_k,
                'repetition_penalty': self.generation.repetition_penalty
            },
            'hardware': {
                'target_gpu': self.hardware.target_gpu,
                'target_vram_gb': self.hardware.target_vram_gb,
                'use_cpu_offload': self.hardware.use_cpu_offload,
                'max_memory_usage': self.hardware.max_memory_usage
            },
            'paths': {
                'cache_dir': self.paths.cache_dir,
                'log_dir': self.paths.log_dir,
                'output_dir': self.paths.output_dir,
                'config_file': self.paths.config_file
            }
        }
        
        try:
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            print(f"✅ 설정 파일 저장: {config_file}")
        except Exception as e:
            print(f"❌ 설정 파일 저장 실패: {e}")
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            self.paths.cache_dir,
            self.paths.log_dir,
            self.paths.output_dir
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                print(f"⚠️ 디렉토리 생성 실패 {directory}: {e}")
    
    def print_config(self):
        """현재 설정 출력"""
        print("📋 현재 설정:")
        print(f"🤖 모델: {self.model.name}")
        print(f"🔧 양자화: {self.quantization.bnb_4bit_quant_type}")
        print(f"🎯 최대 토큰: {self.generation.max_tokens}")
        print(f"🌡️ 온도: {self.generation.temperature}")
        print(f"🎮 대상 GPU: {self.hardware.target_gpu}")
        print(f"💾 VRAM: {self.hardware.target_vram_gb}GB")


# 환경 변수 설정
def setup_environment():
    """환경 변수 설정"""
    # Hugging Face 캐시 디렉토리
    os.environ['HF_HOME'] = './model_cache'
    os.environ['TRANSFORMERS_CACHE'] = './model_cache'
    
    # CUDA 설정 (메모리 최적화)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # 경고 메시지 숨기기
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# 전역 설정 인스턴스
config = Config()

if __name__ == "__main__":
    # 설정 테스트
    setup_environment()
    config.print_config()
    config.save_to_yaml("./config.yaml")
