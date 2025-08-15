# 📚 Korean Bllossom AICA-5B API 참조

이 문서는 Korean Bllossom AICA-5B 양자화 프로젝트의 Python API에 대한 상세한 참조 자료입니다.

## 📋 목차

- [핵심 클래스](#핵심-클래스)
- [설정 관리](#설정-관리)
- [모델 관리](#모델-관리)
- [텍스트 생성](#텍스트-생성)
- [시각-언어 생성](#시각-언어-생성)
- [유틸리티 함수](#유틸리티-함수)
- [예외 처리](#예외-처리)

## 🔧 핵심 클래스

### Config

프로젝트의 전체 설정을 관리하는 클래스입니다.

```python
from config import Config

# 기본 설정으로 초기화
config = Config()

# YAML 파일에서 설정 로드
config = Config("custom_config.yaml")
```

#### 속성

| 속성 | 타입 | 설명 |
|------|------|------|
| `model` | `ModelConfig` | 모델 관련 설정 |
| `quantization` | `QuantizationConfig` | 양자화 설정 |
| `generation` | `GenerationConfig` | 텍스트 생성 설정 |
| `hardware` | `HardwareConfig` | 하드웨어 설정 |
| `paths` | `PathConfig` | 경로 설정 |

#### 메서드

##### `load_from_yaml(config_file: str)`

YAML 파일에서 설정을 로드합니다.

```python
config.load_from_yaml("custom_settings.yaml")
```

##### `save_to_yaml(config_file: str = None)`

현재 설정을 YAML 파일로 저장합니다.

```python
config.save_to_yaml("saved_config.yaml")
```

##### `print_config()`

현재 설정을 콘솔에 출력합니다.

```python
config.print_config()
```

### ModelManager

모델의 로딩, 언로딩, 메모리 관리를 담당하는 클래스입니다.

```python
from model_manager import ModelManager, get_model_manager

# 싱글톤 인스턴스 가져오기 (권장)
manager = get_model_manager(config)

# 직접 생성
manager = ModelManager(config)
```

#### 속성

| 속성 | 타입 | 설명 |
|------|------|------|
| `model` | `MllamaForConditionalGeneration` | 로드된 모델 |
| `processor` | `MllamaProcessor` | 토크나이저 및 프로세서 |
| `device` | `str` | 사용 중인 디바이스 ("cuda" 또는 "cpu") |
| `is_loaded` | `bool` | 모델 로드 상태 |

#### 메서드

##### `load_model() -> bool`

모델과 프로세서를 로드합니다.

```python
success = manager.load_model()
if success:
    print("모델 로드 성공!")
else:
    print("모델 로드 실패!")
```

**반환값**: 로드 성공 여부 (`bool`)

##### `unload_model()`

모델을 메모리에서 언로드합니다.

```python
manager.unload_model()
```

##### `clear_memory()`

GPU 및 시스템 메모리를 정리합니다.

```python
manager.clear_memory()
```

##### `get_model_info() -> dict`

현재 모델 상태 정보를 반환합니다.

```python
info = manager.get_model_info()
print(f"모델 상태: {info['status']}")
print(f"VRAM 사용량: {info.get('vram_allocated', 0):.1f}GB")
```

**반환값**: 모델 정보 딕셔너리

##### `health_check() -> bool`

모델의 정상 작동 여부를 확인합니다.

```python
is_healthy = manager.health_check()
if is_healthy:
    print("모델이 정상 작동 중입니다.")
```

**반환값**: 헬스 체크 통과 여부 (`bool`)

## 📝 텍스트 생성

### TextGenerator

텍스트 생성 기능을 제공하는 클래스입니다.

```python
from text_generator import TextGenerator

generator = TextGenerator(model_manager, config)
```

#### 메서드

##### `generate(prompt, **kwargs) -> dict`

단일 텍스트를 생성합니다.

```python
result = generator.generate(
    prompt="한국의 전통문화에 대해 설명해주세요",
    max_tokens=300,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    do_sample=True,
    system_message="당신은 한국 문화 전문가입니다."
)
```

**매개변수**:

| 매개변수 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `prompt` | `str` | 필수 | 입력 프롬프트 |
| `max_tokens` | `int` | `config.generation.max_tokens` | 최대 생성 토큰 수 |
| `temperature` | `float` | `config.generation.temperature` | 생성 온도 (0.0-2.0) |
| `top_p` | `float` | `config.generation.top_p` | Nucleus sampling 파라미터 |
| `top_k` | `int` | `config.generation.top_k` | Top-k sampling 파라미터 |
| `repetition_penalty` | `float` | `config.generation.repetition_penalty` | 반복 억제 정도 |
| `do_sample` | `bool` | `config.generation.do_sample` | 샘플링 사용 여부 |
| `system_message` | `str` | `None` | 시스템 메시지 |

**반환값**: 생성 결과 딕셔너리

```python
{
    "success": True,
    "response": "생성된 텍스트 내용",
    "full_response": "전체 응답 (입력 포함)",
    "generation_time": 2.34,
    "input_tokens": 15,
    "output_tokens": 87,
    "tokens_per_second": 37.2,
    "parameters": {
        "max_tokens": 300,
        "temperature": 0.7,
        # ... 기타 파라미터
    }
}
```

##### `chat_generate(messages, **kwargs) -> dict`

대화 형식으로 텍스트를 생성합니다.

```python
messages = [
    {"role": "system", "content": "당신은 친근한 AI 어시스턴트입니다."},
    {"role": "user", "content": "안녕하세요!"},
    {"role": "assistant", "content": "안녕하세요! 어떻게 도와드릴까요?"},
    {"role": "user", "content": "오늘 날씨가 어떤가요?"}
]

result = generator.chat_generate(messages, max_tokens=200)
```

**매개변수**:
- `messages`: 대화 히스토리 리스트
- `**kwargs`: `generate()` 메서드와 동일한 파라미터

**반환값**: `generate()` 메서드와 동일한 구조

##### `batch_generate(prompts, **kwargs) -> List[dict]`

여러 프롬프트를 일괄 처리합니다.

```python
prompts = [
    "파이썬의 장점을 설명해주세요",
    "머신러닝이란 무엇인가요?",
    "클라우드 컴퓨팅의 미래는?"
]

results = generator.batch_generate(prompts, max_tokens=200)

for i, result in enumerate(results):
    if result["success"]:
        print(f"질문 {i+1}: {result['response']}")
```

**매개변수**:
- `prompts`: 프롬프트 리스트
- `**kwargs`: 생성 파라미터

**반환값**: 생성 결과 딕셔너리 리스트

### ConversationManager

대화 관리 및 히스토리 추적 클래스입니다.

```python
from text_generator import ConversationManager

conversation = ConversationManager(text_generator, max_history=10)
```

#### 속성

| 속성 | 타입 | 설명 |
|------|------|------|
| `conversation_history` | `List[dict]` | 대화 히스토리 |
| `system_message` | `str` | 시스템 메시지 |
| `max_history` | `int` | 최대 히스토리 길이 |

#### 메서드

##### `set_system_message(message: str)`

시스템 메시지를 설정합니다.

```python
conversation.set_system_message("당신은 전문적인 상담사입니다.")
```

##### `generate_response(user_input, **kwargs) -> dict`

사용자 입력에 대한 응답을 생성합니다.

```python
response = conversation.generate_response(
    "안녕하세요!",
    max_tokens=200,
    temperature=0.7
)

if response["success"]:
    print(f"AI: {response['response']}")
```

##### `clear_history()`

대화 히스토리를 초기화합니다.

```python
conversation.clear_history()
```

##### `get_history_summary() -> dict`

대화 히스토리 요약을 반환합니다.

```python
summary = conversation.get_history_summary()
print(f"총 메시지 수: {summary['total_messages']}")
```

**반환값**:
```python
{
    "total_messages": 12,
    "user_messages": 6,
    "assistant_messages": 6,
    "system_message": "시스템 메시지 내용",
    "latest_messages": [...]  # 최근 4개 메시지
}
```

##### `export_conversation(filename: str = None) -> str`

대화를 텍스트 파일로 내보냅니다.

```python
filename = conversation.export_conversation("my_chat.txt")
print(f"대화 저장됨: {filename}")
```

**반환값**: 저장된 파일명

## 🖼️ 시각-언어 생성

### VisionGenerator

이미지와 텍스트를 함께 처리하는 클래스입니다.

```python
from vision_generator import VisionGenerator

vision_gen = VisionGenerator(model_manager, config)
```

#### 메서드

##### `generate_with_image(image_input, prompt, **kwargs) -> dict`

이미지와 함께 텍스트를 생성합니다.

```python
result = vision_gen.generate_with_image(
    image_input="photo.jpg",  # 또는 URL, PIL Image, bytes
    prompt="이 이미지에 무엇이 보이나요?",
    max_tokens=250,
    temperature=0.1,
    preprocess=True
)
```

**매개변수**:

| 매개변수 | 타입 | 설명 |
|----------|------|------|
| `image_input` | `Union[str, Image.Image, bytes]` | 이미지 입력 |
| `prompt` | `str` | 텍스트 프롬프트 |
| `max_tokens` | `int` | 최대 생성 토큰 수 |
| `temperature` | `float` | 생성 온도 |
| `preprocess` | `bool` | 이미지 전처리 여부 |

**이미지 입력 형식**:
- 로컬 파일 경로: `"./images/photo.jpg"`
- 웹 URL: `"https://example.com/image.png"`
- PIL Image 객체: `Image.open("photo.jpg")`
- 바이트 데이터: `open("image.jpg", "rb").read()`
- Base64 데이터 URL: `"data:image/jpeg;base64,/9j/4AAQ..."`

**반환값**:
```python
{
    "success": True,
    "response": "이미지 분석 결과",
    "full_response": "전체 응답",
    "generation_time": 3.45,
    "input_tokens": 20,
    "output_tokens": 95,
    "tokens_per_second": 27.5,
    "image_size": (1024, 768),
    "parameters": {...}
}
```

##### `describe_image(image_input) -> dict`

이미지를 설명합니다.

```python
result = vision_gen.describe_image("photo.jpg")
print(result['response'])
```

##### `extract_text(image_input) -> dict`

이미지에서 텍스트를 추출합니다 (OCR).

```python
result = vision_gen.extract_text("document.png")
print(f"추출된 텍스트: {result['response']}")
```

##### `analyze_chart(image_input) -> dict`

차트나 그래프를 분석합니다.

```python
result = vision_gen.analyze_chart("sales_chart.png")
print(f"차트 분석: {result['response']}")
```

##### `analyze_table(image_input) -> dict`

표를 분석합니다.

```python
result = vision_gen.analyze_table("data_table.jpg")
print(f"표 분석: {result['response']}")
```

##### `convert_to_markdown(image_input) -> dict`

문서 이미지를 마크다운으로 변환합니다.

```python
result = vision_gen.convert_to_markdown("document.png")
with open("converted.md", "w") as f:
    f.write(result['response'])
```

##### `answer_visual_question(image_input, question) -> dict`

이미지에 대한 구체적인 질문에 답합니다.

```python
result = vision_gen.answer_visual_question(
    "family_photo.jpg",
    "이 사진에 몇 명의 사람이 있나요?"
)
print(f"답변: {result['response']}")
```

##### `batch_analyze_images(image_inputs, prompt, **kwargs) -> List[dict]`

여러 이미지를 일괄 분석합니다.

```python
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = vision_gen.batch_analyze_images(
    images,
    "이 이미지들의 공통점을 찾아주세요"
)
```

### DocumentProcessor

문서 처리 전용 클래스입니다.

```python
from vision_generator import DocumentProcessor

doc_processor = DocumentProcessor(vision_generator)
```

#### 메서드

##### `process_document_page(image_input, task) -> dict`

단일 문서 페이지를 처리합니다.

```python
result = doc_processor.process_document_page(
    "document_page.png",
    task="extract"  # "extract", "summarize", "markdown", "table"
)
```

**작업 유형**:
- `"extract"`: 텍스트 추출
- `"summarize"`: 문서 요약
- `"markdown"`: 마크다운 변환
- `"table"`: 표 데이터 추출

##### `process_multi_page_document(image_inputs, task) -> dict`

다중 페이지 문서를 처리합니다.

```python
pages = ["page1.png", "page2.png", "page3.png"]
result = doc_processor.process_multi_page_document(pages, "markdown")

print(f"처리된 페이지: {result['total_pages']}")
print(f"성공한 페이지: {result['successful_pages']}")
print(f"결합된 내용:\n{result['combined_content']}")
```

**반환값**:
```python
{
    "success": True,
    "page_results": [...],  # 각 페이지 처리 결과
    "combined_content": "전체 결합된 내용",
    "total_pages": 3,
    "successful_pages": 3
}
```

## 🛠️ 유틸리티 함수

### 설정 관련

#### `setup_environment()`

환경 변수를 설정합니다.

```python
from config import setup_environment

setup_environment()
```

설정되는 환경 변수:
- `HF_HOME`: Hugging Face 캐시 디렉토리
- `TRANSFORMERS_CACHE`: Transformers 캐시 디렉토리
- `PYTORCH_CUDA_ALLOC_CONF`: CUDA 메모리 설정
- `TOKENIZERS_PARALLELISM`: 토크나이저 병렬 처리 설정

### 모델 관리

#### `get_model_manager(config) -> ModelManager`

싱글톤 모델 매니저 인스턴스를 반환합니다.

```python
from model_manager import get_model_manager

manager = get_model_manager(config)
```

## ⚠️ 예외 처리

### 일반적인 예외

#### `torch.cuda.OutOfMemoryError`

VRAM 부족 시 발생하는 예외입니다.

```python
try:
    result = generator.generate("긴 프롬프트...")
except torch.cuda.OutOfMemoryError:
    print("VRAM 부족! 메모리를 정리하고 다시 시도하세요.")
    manager.clear_memory()
```

#### `ValueError`

잘못된 매개변수 사용 시 발생합니다.

```python
try:
    result = generator.generate("")  # 빈 프롬프트
except ValueError as e:
    print(f"매개변수 오류: {e}")
```

#### `ConnectionError`

네트워크 관련 오류입니다.

```python
try:
    result = vision_gen.describe_image("https://broken-url.com/image.jpg")
except requests.ConnectionError:
    print("이미지 URL에 접근할 수 없습니다.")
```

### 안전한 호출 패턴

#### 재시도 메커니즘

```python
import time

def safe_generate(generator, prompt, max_retries=3):
    """안전한 생성 함수 (재시도 포함)"""
    
    for attempt in range(max_retries):
        try:
            result = generator.generate(prompt)
            if result["success"]:
                return result
            else:
                print(f"시도 {attempt + 1} 실패: {result['error']}")
                
        except Exception as e:
            print(f"시도 {attempt + 1} 예외: {e}")
            
        # 재시도 전 대기 및 메모리 정리
        time.sleep(2)
        generator.model_manager.clear_memory()
    
    return {"success": False, "error": "모든 재시도 실패"}

# 사용 예시
result = safe_generate(generator, "복잡한 질문")
```

#### 컨텍스트 매니저

```python
from contextlib import contextmanager

@contextmanager
def model_context(config):
    """모델 자동 관리 컨텍스트"""
    manager = get_model_manager(config)
    
    try:
        if not manager.is_loaded:
            success = manager.load_model()
            if not success:
                raise RuntimeError("모델 로드 실패")
        
        yield manager
        
    finally:
        manager.clear_memory()

# 사용 예시
with model_context(config) as manager:
    generator = TextGenerator(manager, config)
    result = generator.generate("테스트 프롬프트")
    print(result['response'])
```

## 🔧 고급 사용법

### 커스텀 설정 클래스

```python
from config import Config
from dataclasses import dataclass

@dataclass
class CustomConfig(Config):
    """사용자 정의 설정 클래스"""
    
    def __init__(self):
        super().__init__()
        # 성능 최적화 설정
        self.generation.max_tokens = 200
        self.generation.temperature = 0.5
        self.hardware.max_memory_usage = 0.8
    
    def set_creative_mode(self):
        """창의적 모드 설정"""
        self.generation.temperature = 0.9
        self.generation.top_p = 0.95
        self.generation.do_sample = True
    
    def set_precise_mode(self):
        """정확한 모드 설정"""
        self.generation.temperature = 0.1
        self.generation.do_sample = False

# 사용 예시
custom_config = CustomConfig()
custom_config.set_creative_mode()
```

### 성능 모니터링 데코레이터

```python
import time
import functools

def monitor_performance(func):
    """성능 모니터링 데코레이터"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            result = func(*args, **kwargs)
            
            # 성능 정보 추가
            if isinstance(result, dict) and "success" in result:
                end_time = time.time()
                end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                result["performance"] = {
                    "execution_time": end_time - start_time,
                    "memory_used": (end_memory - start_memory) / 1024**3,
                    "function_name": func.__name__
                }
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"함수 {func.__name__} 실행 실패 ({execution_time:.2f}초): {e}")
            raise
    
    return wrapper

# 사용 예시
@monitor_performance
def generate_with_monitoring(generator, prompt):
    return generator.generate(prompt)

result = generate_with_monitoring(generator, "테스트")
print(f"실행 시간: {result['performance']['execution_time']:.2f}초")
```

### 비동기 처리

```python
import asyncio
import concurrent.futures

async def async_batch_generate(generator, prompts, max_workers=2):
    """비동기 배치 생성"""
    
    def generate_single(prompt):
        return generator.generate(prompt)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, generate_single, prompt)
            for prompt in prompts
        ]
        
        results = await asyncio.gather(*tasks)
        return results

# 사용 예시
async def main():
    prompts = ["질문 1", "질문 2", "질문 3"]
    results = await async_batch_generate(generator, prompts)
    
    for i, result in enumerate(results):
        print(f"결과 {i+1}: {result['response']}")

# asyncio.run(main())
```

---

이 API 참조 문서를 통해 Korean Bllossom AICA-5B 프로젝트의 모든 기능을 효과적으로 활용하실 수 있습니다. 추가 질문이나 예제가 필요하시면 GitHub Issues를 통해 문의해주세요.