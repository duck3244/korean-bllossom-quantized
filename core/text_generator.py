# text_generator.py
# 텍스트 생성 기능

import torch
import time
from typing import List, Dict, Optional, Any
from core.model_manager import ModelManager
from core.config import Config
import logging

logger = logging.getLogger(__name__)

class TextGenerator:
    """텍스트 생성 클래스"""
    
    def __init__(self, model_manager: ModelManager, config: Config):
        self.model_manager = model_manager
        self.config = config
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            max_tokens: 최대 생성 토큰 수
            temperature: 생성 온도 (0.0-2.0)
            top_p: nucleus sampling 파라미터
            top_k: top-k sampling 파라미터
            repetition_penalty: 반복 억제 정도
            do_sample: 샘플링 여부
            system_message: 시스템 메시지
            
        Returns:
            생성 결과 딕셔너리
        """
        
        if not self.model_manager.is_loaded:
            return {
                "success": False,
                "error": "모델이 로드되지 않았습니다.",
                "response": None
            }
        
        # 기본값 설정
        max_tokens = max_tokens or self.config.generation.max_tokens
        temperature = temperature or self.config.generation.temperature
        top_p = top_p or self.config.generation.top_p
        top_k = top_k or self.config.generation.top_k
        repetition_penalty = repetition_penalty or self.config.generation.repetition_penalty
        do_sample = do_sample if do_sample is not None else self.config.generation.do_sample
        
        print(f"💬 입력: {prompt}")
        print("🤖 생성 중...")
        
        start_time = time.time()
        
        try:
            # 메시지 구성
            messages = []
            
            if system_message:
                messages.append({
                    'role': 'system', 
                    'content': [{'type': 'text', 'text': system_message}]
                })
            
            messages.append({
                'role': 'user', 
                'content': [{'type': 'text', 'text': prompt}]
            })
            
            # 입력 텍스트 처리
            input_text = self.model_manager.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 토크나이징
            inputs = self.model_manager.processor(
                images=None,  # 텍스트 생성 모드
                text=input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.model_manager.device)
            
            # 생성 파라미터 설정
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "pad_token_id": self.model_manager.processor.tokenizer.eos_token_id,
                "eos_token_id": self.model_manager.processor.tokenizer.convert_tokens_to_ids('<|eot_id|>'),
                "use_cache": True
            }
            
            if do_sample:
                generation_kwargs.update({
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty
                })
            
            # 텍스트 생성
            with torch.no_grad():
                output = self.model_manager.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # 디코딩
            full_response = self.model_manager.processor.decode(output[0], skip_special_tokens=True)
            
            # 응답 추출
            clean_response = self._extract_response(full_response, input_text)
            
            generation_time = time.time() - start_time
            input_tokens = len(inputs['input_ids'][0])
            output_tokens = len(output[0]) - input_tokens
            tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
            
            print(f"✨ 응답: {clean_response}")
            print(f"⏱️ 생성 시간: {generation_time:.2f}초")
            print(f"🚀 속도: {tokens_per_second:.1f} 토큰/초")
            
            return {
                "success": True,
                "response": clean_response,
                "full_response": full_response,
                "generation_time": generation_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens_per_second": tokens_per_second,
                "parameters": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "do_sample": do_sample
                }
            }
            
        except torch.cuda.OutOfMemoryError:
            self.model_manager.clear_memory()
            error_msg = "VRAM 부족! 더 짧은 프롬프트나 적은 토큰 수를 사용해보세요."
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "response": None
            }
            
        except Exception as e:
            error_msg = f"생성 실패: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(f"Text generation failed: {e}")
            return {
                "success": False,
                "error": error_msg,
                "response": None
            }
    
    def _extract_response(self, full_response: str, input_text: str) -> str:
        """응답에서 실제 생성된 부분만 추출"""
        try:
            # assistant 태그 이후의 내용 추출
            if "assistant\n\n" in full_response:
                response_start = full_response.find("assistant\n\n") + len("assistant\n\n")
                clean_response = full_response[response_start:].strip()
            elif "assistant" in full_response:
                clean_response = full_response.split("assistant")[-1].strip()
            else:
                # 입력 부분 제거
                clean_response = full_response.replace(input_text, "").strip()
            
            # 특수 토큰 정리
            clean_response = clean_response.replace('<|eot_id|>', '').strip()
            
            return clean_response
            
        except Exception as e:
            logger.warning(f"Response extraction failed: {e}")
            return full_response
    
    def chat_generate(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        대화 형식으로 텍스트 생성
        
        Args:
            messages: 대화 히스토리 [{"role": "user/assistant", "content": "..."}]
            **kwargs: 생성 파라미터
            
        Returns:
            생성 결과 딕셔너리
        """
        
        if not self.model_manager.is_loaded:
            return {
                "success": False,
                "error": "모델이 로드되지 않았습니다.",
                "response": None
            }
        
        try:
            # 메시지를 프롬프트로 변환
            formatted_messages = []
            for msg in messages:
                if isinstance(msg["content"], str):
                    content = [{"type": "text", "text": msg["content"]}]
                else:
                    content = msg["content"]
                
                formatted_messages.append({
                    "role": msg["role"],
                    "content": content
                })
            
            # 입력 텍스트 처리
            input_text = self.model_manager.processor.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 토크나이징
            inputs = self.model_manager.processor(
                images=None,
                text=input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.model_manager.device)
            
            # 기본 파라미터 설정
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_tokens", self.config.generation.max_tokens),
                "temperature": kwargs.get("temperature", self.config.generation.temperature),
                "do_sample": kwargs.get("do_sample", self.config.generation.do_sample),
                "pad_token_id": self.model_manager.processor.tokenizer.eos_token_id,
                "eos_token_id": self.model_manager.processor.tokenizer.convert_tokens_to_ids('<|eot_id|>'),
                "use_cache": True
            }
            
            if generation_kwargs["do_sample"]:
                generation_kwargs.update({
                    "top_p": kwargs.get("top_p", self.config.generation.top_p),
                    "top_k": kwargs.get("top_k", self.config.generation.top_k),
                    "repetition_penalty": kwargs.get("repetition_penalty", self.config.generation.repetition_penalty)
                })
            
            start_time = time.time()
            
            # 생성
            with torch.no_grad():
                output = self.model_manager.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # 결과 처리
            full_response = self.model_manager.processor.decode(output[0], skip_special_tokens=True)
            clean_response = self._extract_response(full_response, input_text)
            
            generation_time = time.time() - start_time
            input_tokens = len(inputs['input_ids'][0])
            output_tokens = len(output[0]) - input_tokens
            tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
            
            return {
                "success": True,
                "response": clean_response,
                "full_response": full_response,
                "generation_time": generation_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens_per_second": tokens_per_second,
                "parameters": generation_kwargs
            }
            
        except Exception as e:
            error_msg = f"대화 생성 실패: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "response": None
            }
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        배치 텍스트 생성
        
        Args:
            prompts: 프롬프트 리스트
            **kwargs: 생성 파라미터
            
        Returns:
            생성 결과 리스트
        """
        results = []
        
        print(f"📦 배치 생성 시작: {len(prompts)}개 프롬프트")
        
        for i, prompt in enumerate(prompts):
            print(f"🔄 처리 중: {i+1}/{len(prompts)}")
            result = self.generate(prompt, **kwargs)
            results.append(result)
            
            # 메모리 정리
            if i % 5 == 4:  # 5개마다 메모리 정리
                self.model_manager.clear_memory()
        
        print("✅ 배치 생성 완료!")
        return results
    
    def stream_generate(self, prompt: str, **kwargs):
        """
        스트리밍 텍스트 생성 (향후 구현)
        
        Args:
            prompt: 입력 프롬프트
            **kwargs: 생성 파라미터
            
        Yields:
            생성된 토큰들
        """
        # TODO: 스트리밍 생성 구현
        # 현재는 일반 생성으로 대체
        result = self.generate(prompt, **kwargs)
        if result["success"]:
            yield result["response"]
        else:
            yield f"오류: {result['error']}"


class ConversationManager:
    """대화 관리 클래스"""
    
    def __init__(self, text_generator: TextGenerator, max_history: int = 10):
        self.text_generator = text_generator
        self.max_history = max_history
        self.conversation_history = []
        self.system_message = None
    
    def set_system_message(self, message: str):
        """시스템 메시지 설정"""
        self.system_message = message
    
    def add_message(self, role: str, content: str):
        """대화 히스토리에 메시지 추가"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # 히스토리 길이 제한
        if len(self.conversation_history) > self.max_history * 2:  # user + assistant 쌍
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
    
    def generate_response(self, user_input: str, **kwargs) -> Dict[str, Any]:
        """사용자 입력에 대한 응답 생성"""
        
        # 사용자 메시지 추가
        self.add_message("user", user_input)
        
        # 대화 히스토리 구성
        messages = []
        
        if self.system_message:
            messages.append({
                "role": "system",
                "content": self.system_message
            })
        
        # 히스토리 추가
        for msg in self.conversation_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # 응답 생성
        result = self.text_generator.chat_generate(messages, **kwargs)
        
        if result["success"]:
            # 어시스턴트 응답 추가
            self.add_message("assistant", result["response"])
        
        return result
    
    def clear_history(self):
        """대화 히스토리 초기화"""
        self.conversation_history = []
        print("🗑️ 대화 히스토리가 초기화되었습니다.")
    
    def get_history_summary(self) -> Dict[str, Any]:
        """대화 히스토리 요약 반환"""
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len([msg for msg in self.conversation_history if msg["role"] == "user"]),
            "assistant_messages": len([msg for msg in self.conversation_history if msg["role"] == "assistant"]),
            "system_message": self.system_message,
            "latest_messages": self.conversation_history[-4:] if self.conversation_history else []
        }
    
    def export_conversation(self, filename: str = None) -> str:
        """대화 히스토리를 파일로 내보내기"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=== Korean Bllossom 대화 기록 ===\n\n")
                
                if self.system_message:
                    f.write(f"시스템 메시지: {self.system_message}\n\n")
                
                for msg in self.conversation_history:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg["timestamp"]))
                    role_name = "사용자" if msg["role"] == "user" else "어시스턴트"
                    f.write(f"[{timestamp}] {role_name}: {msg['content']}\n\n")
            
            print(f"📁 대화가 저장되었습니다: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ 대화 저장 실패: {e}")
            return None


if __name__ == "__main__":
    # 텍스트 생성기 테스트
    from config import config
    from model_manager import get_model_manager
    
    # 모델 매니저 초기화
    manager = get_model_manager(config)
    
    if manager.load_model():
        # 텍스트 생성기 초기화
        generator = TextGenerator(manager, config)
        
        # 단일 생성 테스트
        result = generator.generate("안녕하세요! 오늘 날씨가 어떤가요?")
        print(f"생성 결과: {result}")
        
        # 대화 관리자 테스트
        conversation = ConversationManager(generator)
        conversation.set_system_message("당신은 친근하고 도움이 되는 AI 어시스턴트입니다.")
        
        # 대화 테스트
        response1 = conversation.generate_response("안녕하세요!")
        print(f"응답 1: {response1}")
        
        response2 = conversation.generate_response("오늘 뭐 하고 계세요?")
        print(f"응답 2: {response2}")
        
        # 히스토리 요약
        summary = conversation.get_history_summary()
        print(f"대화 요약: {summary}")
        
        # 대화 저장
        conversation.export_conversation()
        
        manager.unload_model()
    else:
        print("모델 로드 실패!")