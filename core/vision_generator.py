# vision_generator.py
# 시각-언어 모델 기능

import torch
import time
from PIL import Image
import requests
import base64
import io
from typing import Dict, Any, Optional, Union, List
from core.model_manager import ModelManager
from core.config import Config
import logging

logger = logging.getLogger(__name__)

class VisionGenerator:
    """시각-언어 생성 클래스"""
    
    def __init__(self, model_manager: ModelManager, config: Config):
        self.model_manager = model_manager
        self.config = config
        self.supported_formats = ['JPEG', 'PNG', 'WebP', 'BMP']
    
    def _load_image(self, image_input: Union[str, Image.Image, bytes]) -> Optional[Image.Image]:
        """다양한 형태의 이미지 입력을 PIL Image로 변환"""
        try:
            if isinstance(image_input, Image.Image):
                return image_input
            
            elif isinstance(image_input, str):
                if image_input.startswith(('http://', 'https://')):
                    # URL에서 이미지 로드
                    response = requests.get(image_input, stream=True, timeout=10)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content))
                elif image_input.startswith('data:image'):
                    # Base64 데이터 URL
                    header, data = image_input.split(',', 1)
                    image_bytes = base64.b64decode(data)
                    image = Image.open(io.BytesIO(image_bytes))
                else:
                    # 로컬 파일 경로
                    image = Image.open(image_input)
            
            elif isinstance(image_input, bytes):
                # 바이트 데이터
                image = Image.open(io.BytesIO(image_input))
            
            else:
                raise ValueError(f"지원하지 않는 이미지 입력 형태: {type(image_input)}")
            
            # 이미지 포맷 확인
            if image.format not in self.supported_formats:
                print(f"⚠️ 경고: {image.format} 포맷은 지원되지 않을 수 있습니다.")
            
            # RGB로 변환 (필요한 경우)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            print(f"❌ 이미지 로드 실패: {e}")
            return None
    
    def _preprocess_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """이미지 전처리 (크기 조정 등)"""
        try:
            # 이미지 크기 확인 및 조정
            width, height = image.size
            
            if max(width, height) > max_size:
                # 비율 유지하면서 크기 조정
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"🖼️ 이미지 크기 조정: {width}x{height} → {new_width}x{new_height}")
            
            return image
            
        except Exception as e:
            print(f"⚠️ 이미지 전처리 실패: {e}")
            return image
    
    def generate_with_image(
        self,
        image_input: Union[str, Image.Image, bytes],
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        이미지와 함께 텍스트 생성
        
        Args:
            image_input: 이미지 (URL, 파일경로, PIL Image, 바이트)
            prompt: 텍스트 프롬프트
            max_tokens: 최대 생성 토큰 수
            temperature: 생성 온도
            preprocess: 이미지 전처리 여부
            
        Returns:
            생성 결과 딕셔너리
        """
        
        if not self.model_manager.is_loaded:
            return {
                "success": False,
                "error": "모델이 로드되지 않았습니다.",
                "response": None
            }
        
        print(f"🖼️ 이미지와 함께 생성 시작")
        print(f"💬 프롬프트: {prompt}")
        
        start_time = time.time()
        
        try:
            # 이미지 로드
            image = self._load_image(image_input)
            if image is None:
                return {
                    "success": False,
                    "error": "이미지 로드에 실패했습니다.",
                    "response": None
                }
            
            # 이미지 전처리
            if preprocess:
                image = self._preprocess_image(image)
            
            print(f"📐 이미지 크기: {image.size}")
            
            # 기본값 설정
            max_tokens = max_tokens or self.config.generation.max_tokens
            temperature = temperature or 0.1  # 이미지 분석은 낮은 온도 권장
            
            # 메시지 구성
            messages = [
                {'role': 'user', 'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt}
                ]},
            ]
            
            # 입력 텍스트 처리
            input_text = self.model_manager.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 토크나이징 (이미지 포함)
            inputs = self.model_manager.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.model_manager.device)
            
            print("🤖 이미지 분석 및 생성 중...")
            
            # 생성
            with torch.no_grad():
                output = self.model_manager.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    eos_token_id=self.model_manager.processor.tokenizer.convert_tokens_to_ids('<|eot_id|>'),
                    use_cache=False,  # 메모리 절약
                    pad_token_id=self.model_manager.processor.tokenizer.eos_token_id
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
                "image_size": image.size,
                "parameters": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "preprocess": preprocess
                }
            }
            
        except torch.cuda.OutOfMemoryError:
            self.model_manager.clear_memory()
            error_msg = "VRAM 부족! 이미지 처리는 더 많은 메모리가 필요합니다."
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "response": None
            }
            
        except Exception as e:
            error_msg = f"이미지 생성 실패: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(f"Vision generation failed: {e}")
            return {
                "success": False,
                "error": error_msg,
                "response": None
            }
    
    def _extract_response(self, full_response: str, input_text: str) -> str:
        """응답에서 실제 생성된 부분만 추출"""
        try:
            if "assistant\n\n" in full_response:
                response_start = full_response.find("assistant\n\n") + len("assistant\n\n")
                clean_response = full_response[response_start:].strip()
            elif "assistant" in full_response:
                clean_response = full_response.split("assistant")[-1].strip()
            else:
                clean_response = full_response.replace(input_text, "").strip()
            
            # 특수 토큰 정리
            clean_response = clean_response.replace('<|eot_id|>', '').strip()
            
            return clean_response
            
        except Exception as e:
            logger.warning(f"Response extraction failed: {e}")
            return full_response
    
    def describe_image(self, image_input: Union[str, Image.Image, bytes]) -> Dict[str, Any]:
        """이미지 설명 생성"""
        return self.generate_with_image(
            image_input,
            "이 이미지에 무엇이 보이나요? 자세히 설명해주세요.",
            max_tokens=200,
            temperature=0.1
        )
    
    def extract_text(self, image_input: Union[str, Image.Image, bytes]) -> Dict[str, Any]:
        """이미지에서 텍스트 추출 (OCR)"""
        return self.generate_with_image(
            image_input,
            "이 이미지에 있는 텍스트를 모두 추출해주세요. 텍스트만 정확히 적어주세요.",
            max_tokens=300,
            temperature=0.0
        )
    
    def analyze_chart(self, image_input: Union[str, Image.Image, bytes]) -> Dict[str, Any]:
        """차트/그래프 분석"""
        return self.generate_with_image(
            image_input,
            "이 차트나 그래프를 분석해주세요. 주요 데이터, 트렌드, 패턴을 설명해주세요.",
            max_tokens=300,
            temperature=0.1
        )
    
    def analyze_table(self, image_input: Union[str, Image.Image, bytes]) -> Dict[str, Any]:
        """표 분석"""
        return self.generate_with_image(
            image_input,
            "이 표의 내용을 분석하고 정리해주세요. 중요한 정보와 패턴을 설명해주세요.",
            max_tokens=400,
            temperature=0.1
        )
    
    def convert_to_markdown(self, image_input: Union[str, Image.Image, bytes]) -> Dict[str, Any]:
        """문서 이미지를 마크다운으로 변환"""
        return self.generate_with_image(
            image_input,
            "이 문서를 마크다운 형식으로 변환해주세요. 구조와 형식을 유지하면서 정확히 변환해주세요.",
            max_tokens=500,
            temperature=0.0
        )
    
    def answer_visual_question(
        self, 
        image_input: Union[str, Image.Image, bytes], 
        question: str
    ) -> Dict[str, Any]:
        """이미지에 대한 질문 답변"""
        return self.generate_with_image(
            image_input,
            f"질문: {question}\n\n이 이미지를 보고 위 질문에 답해주세요.",
            max_tokens=250,
            temperature=0.1
        )
    
    def batch_analyze_images(
        self, 
        image_inputs: List[Union[str, Image.Image, bytes]], 
        prompt: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """여러 이미지 일괄 분석"""
        results = []
        
        print(f"📦 배치 이미지 분석 시작: {len(image_inputs)}개 이미지")
        
        for i, image_input in enumerate(image_inputs):
            print(f"🔄 처리 중: {i+1}/{len(image_inputs)}")
            
            try:
                result = self.generate_with_image(image_input, prompt, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"❌ 이미지 {i+1} 처리 실패: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "response": None,
                    "image_index": i
                })
            
            # 메모리 정리
            if i % 3 == 2:  # 3개마다 메모리 정리
                self.model_manager.clear_memory()
        
        print("✅ 배치 이미지 분석 완료!")
        return results


class DocumentProcessor:
    """문서 처리 전용 클래스"""
    
    def __init__(self, vision_generator: VisionGenerator):
        self.vision_generator = vision_generator
    
    def process_document_page(
        self, 
        image_input: Union[str, Image.Image, bytes],
        task: str = "extract"
    ) -> Dict[str, Any]:
        """
        문서 페이지 처리
        
        Args:
            image_input: 문서 이미지
            task: 처리 작업 ("extract", "summarize", "markdown", "table")
        """
        
        prompts = {
            "extract": "이 문서의 모든 텍스트를 정확히 추출해주세요.",
            "summarize": "이 문서의 내용을 요약해주세요. 주요 포인트만 간결하게 정리해주세요.",
            "markdown": "이 문서를 마크다운 형식으로 변환해주세요. 제목, 목록, 표 등의 구조를 유지해주세요.",
            "table": "이 문서에 있는 표의 데이터를 추출하고 정리해주세요."
        }
        
        prompt = prompts.get(task, prompts["extract"])
        max_tokens = 500 if task in ["markdown", "table"] else 300
        
        return self.vision_generator.generate_with_image(
            image_input,
            prompt,
            max_tokens=max_tokens,
            temperature=0.0
        )
    
    def process_multi_page_document(
        self, 
        image_inputs: List[Union[str, Image.Image, bytes]],
        task: str = "extract"
    ) -> Dict[str, Any]:
        """다중 페이지 문서 처리"""
        
        print(f"📄 다중 페이지 문서 처리 시작: {len(image_inputs)}페이지")
        
        page_results = []
        combined_content = []
        
        for i, image_input in enumerate(image_inputs):
            print(f"📃 페이지 {i+1}/{len(image_inputs)} 처리 중...")
            
            result = self.process_document_page(image_input, task)
            page_results.append(result)
            
            if result["success"]:
                combined_content.append(f"=== 페이지 {i+1} ===\n{result['response']}\n")
            else:
                combined_content.append(f"=== 페이지 {i+1} ===\n오류: {result['error']}\n")
        
        # 전체 내용 결합
        full_content = "\n".join(combined_content)
        
        return {
            "success": True,
            "page_results": page_results,
            "combined_content": full_content,
            "total_pages": len(image_inputs),
            "successful_pages": len([r for r in page_results if r["success"]])
        }


if __name__ == "__main__":
    # 시각-언어 생성기 테스트
    from config import config
    from model_manager import get_model_manager
    
    # 모델 매니저 초기화
    manager = get_model_manager(config)
    
    if manager.load_model():
        # 시각-언어 생성기 초기화
        vision_gen = VisionGenerator(manager, config)
        
        # 테스트 이미지 URL
        test_image = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/1200px-React-icon.svg.png"
        
        # 이미지 설명 테스트
        result = vision_gen.describe_image(test_image)
        print(f"이미지 설명: {result}")
        
        # 시각적 질문 답변 테스트
        qa_result = vision_gen.answer_visual_question(
            test_image, 
            "이 로고는 어떤 기술을 나타내나요?"
        )
        print(f"질문 답변: {qa_result}")
        
        # 문서 처리기 테스트
        doc_processor = DocumentProcessor(vision_gen)
        
        manager.unload_model()
    else:
        print("모델 로드 실패!")