# model_manager.py
# 모델 로딩 및 관리

import torch
import gc
import psutil
from transformers import (
    MllamaForConditionalGeneration, 
    MllamaProcessor,
    BitsAndBytesConfig
)
from core.config import Config
import logging
import time
from typing import Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """모델 로딩 및 관리 클래스"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        
        # 시스템 정보 확인
        self._check_system_requirements()
    
    def _check_system_requirements(self):
        """시스템 요구사항 확인"""
        print("🔍 시스템 요구사항 확인 중...")
        
        # CPU 정보
        cpu_count = psutil.cpu_count()
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💻 CPU: {cpu_count}코어")
        print(f"🧠 RAM: {ram_gb:.1f}GB")
        
        # GPU 정보
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU: {gpu_name}")
            print(f"💾 VRAM: {total_vram:.1f}GB")
            
            # 권장사항 확인
            if total_vram < 6:
                print("⚠️ 경고: VRAM이 6GB 미만입니다. 성능 저하가 예상됩니다.")
            elif total_vram >= 8:
                print("✅ VRAM이 충분합니다.")
        else:
            print("❌ CUDA GPU를 찾을 수 없습니다.")
            print("CPU 모드로 실행됩니다. (매우 느림)")
    
    def _setup_quantization_config(self) -> BitsAndBytesConfig:
        """양자화 설정 생성"""
        return BitsAndBytesConfig(
            load_in_4bit=self.config.quantization.load_in_4bit,
            bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.config.quantization.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.config.quantization.bnb_4bit_use_double_quant,
        )
    
    def load_model(self) -> bool:
        """모델과 프로세서 로드"""
        if self.is_loaded:
            print("✅ 모델이 이미 로드되어 있습니다.")
            return True
        
        print("🚀 Korean Bllossom AICA-5B 모델 로딩 중...")
        print(f"📱 사용 디바이스: {self.device}")
        
        start_time = time.time()
        
        try:
            # 메모리 정리
            self.clear_memory()
            
            # 양자화 설정
            quantization_config = self._setup_quantization_config()
            
            # 모델 로드
            print("📦 모델 가중치 다운로드 중...")
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.config.model.name,
                quantization_config=quantization_config,
                device_map=self.config.model.device_map,
                torch_dtype=getattr(torch, self.config.model.torch_dtype),
                low_cpu_mem_usage=self.config.model.low_cpu_mem_usage,
                trust_remote_code=self.config.model.trust_remote_code,
                cache_dir=self.config.paths.cache_dir
            )
            
            # 프로세서 로드
            print("🔧 프로세서 로딩 중...")
            self.processor = MllamaProcessor.from_pretrained(
                self.config.model.name,
                trust_remote_code=self.config.model.trust_remote_code,
                cache_dir=self.config.paths.cache_dir
            )
            
            self.is_loaded = True
            load_time = time.time() - start_time
            
            print(f"✅ 모델 로딩 완료! ({load_time:.1f}초)")
            
            # 메모리 사용량 확인
            self._print_memory_usage()
            
            return True
            
        except torch.cuda.OutOfMemoryError:
            print("❌ VRAM 부족으로 모델 로딩 실패!")
            print("💡 해결책:")
            print("1. 다른 GPU 사용 프로그램들을 종료하세요")
            print("2. 시스템을 재부팅하세요")
            print("3. 더 작은 모델을 사용하세요")
            self.clear_memory()
            return False
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {str(e)}")
            logger.error(f"Model loading failed: {e}")
            return False
    
    def unload_model(self):
        """모델 언로드"""
        if not self.is_loaded:
            print("모델이 로드되지 않은 상태입니다.")
            return
        
        print("📤 모델 언로드 중...")
        
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            self.is_loaded = False
            self.clear_memory()
            
            print("✅ 모델 언로드 완료!")
            
        except Exception as e:
            print(f"⚠️ 모델 언로드 중 오류: {e}")
    
    def clear_memory(self):
        """메모리 정리"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _print_memory_usage(self):
        """메모리 사용량 출력"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"📊 VRAM 사용량:")
            print(f"   할당됨: {allocated:.1f}GB")
            print(f"   예약됨: {reserved:.1f}GB")
            print(f"   전체: {total:.1f}GB")
            print(f"   사용률: {(allocated/total)*100:.1f}%")
            
            # 경고 확인
            usage_ratio = allocated / total
            if usage_ratio > 0.9:
                print("⚠️ 경고: VRAM 사용률이 90%를 초과했습니다.")
            elif usage_ratio > 0.8:
                print("⚠️ 주의: VRAM 사용률이 80%를 초과했습니다.")
        
        # 시스템 RAM 사용량
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024**3)
        ram_total_gb = ram.total / (1024**3)
        print(f"🧠 RAM 사용량: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB ({ram.percent:.1f}%)")
    
    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        info = {
            "status": "loaded",
            "model_name": self.config.model.name,
            "device": self.device,
            "quantization": self.config.quantization.bnb_4bit_quant_type,
            "torch_dtype": self.config.model.torch_dtype
        }
        
        if torch.cuda.is_available():
            info["vram_allocated"] = torch.cuda.memory_allocated(0) / (1024**3)
            info["vram_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info
    
    def health_check(self) -> bool:
        """모델 상태 확인"""
        if not self.is_loaded:
            return False
        
        try:
            # 간단한 테스트 생성
            test_messages = [
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': '안녕하세요'}
                ]},
            ]
            
            input_text = self.processor.apply_chat_template(
                test_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                images=None,
                text=input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.device)
            
            # 매우 짧은 생성으로 테스트
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def optimize_for_inference(self):
        """추론 최적화"""
        if not self.is_loaded:
            print("모델이 로드되지 않았습니다.")
            return
        
        print("⚡ 추론 최적화 적용 중...")
        
        try:
            # 모델을 evaluation 모드로 설정
            self.model.eval()
            
            # 그래디언트 계산 비활성화
            for param in self.model.parameters():
                param.requires_grad = False
            
            # CUDA 최적화 (가능한 경우)
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            
            print("✅ 추론 최적화 완료!")
            
        except Exception as e:
            print(f"⚠️ 최적화 중 오류: {e}")


# 싱글톤 패턴으로 모델 매니저 인스턴스 관리
_model_manager_instance = None

def get_model_manager(config: Optional[Config] = None) -> ModelManager:
    """모델 매니저 싱글톤 인스턴스 반환"""
    global _model_manager_instance
    
    if _model_manager_instance is None:
        if config is None:
            from config import config as default_config
            config = default_config
        _model_manager_instance = ModelManager(config)
    
    return _model_manager_instance


if __name__ == "__main__":
    # 모델 매니저 테스트
    from config import config
    
    manager = ModelManager(config)
    
    # 모델 로드 테스트
    if manager.load_model():
        print("모델 로드 성공!")
        
        # 헬스 체크
        if manager.health_check():
            print("✅ 모델이 정상 작동합니다.")
        else:
            print("❌ 모델에 문제가 있습니다.")
        
        # 모델 정보 출력
        info = manager.get_model_info()
        print(f"📋 모델 정보: {info}")
        
        # 최적화 적용
        manager.optimize_for_inference()
        
        # 언로드
        manager.unload_model()
    else:
        print("모델 로드 실패!")
