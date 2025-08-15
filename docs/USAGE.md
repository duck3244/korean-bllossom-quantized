# 📖 Korean Bllossom AICA-5B 사용법 가이드

이 문서는 Korean Bllossom AICA-5B 양자화 프로젝트의 상세한 사용법을 설명합니다.

## 🚀 시작하기

### 기본 실행 순서

```bash
# 1. 가상환경 활성화
source bllossom_env/bin/activate

# 2. 시스템 상태 확인
python main.py --check

# 3. 첫 실행 (데모 모드)
python main.py --demo
```

## 🎯 실행 모드

### 1. 대화형 메뉴 모드

가장 사용자 친화적인 방법입니다.

```bash
python main.py
```

**메뉴 옵션**:
- `1`: 데모 모드 (모든 기능 테스트)
- `2`: 간단 채팅 모드
- `3`: CLI 모드 안내
- `4`: 설정 확인
- `5`: 시스템 정보
- `0`: 종료

### 2. 직접 실행 모드

특정 기능을 바로 실행할 수 있습니다.

```bash
# 데모 모드
python main.py --demo

# 채팅 모드
python main.py --chat

# 시스템 확인
python main.py --check

# 설정 정보
python main.py --config
```

## 💬 텍스트 생성 사용법

### CLI를 통한 텍스트 생성

#### 단일 텍스트 생성

```bash
# 기본 사용
python cli_interface.py text "한국의 전통문화에 대해 설명해주세요"

# 파라미터 조정
python cli_interface.py text "창의적인 이야기를 써주세요" \
  --max-tokens 400 \
  --temperature 0.9 \
  --top-p 0.95
```

#### 대화형 모드

```bash
# 대화형 채팅 시작
python cli_interface.py text --interactive

# 시스템 메시지와 함께 시작
python cli_interface.py text --interactive \
  --system-message "당신은 전문적인 한국어 교사입니다."
```

**대화형 모드 명령어**:
- `/help`: 도움말 표시
- `/clear`: 대화 히스토리 초기화
- `/save`: 대화를 파일로 저장
- `/stats`: 모델 상태 정보
- `/quit`: 종료

### Python API 사용

#### 기본 텍스트 생성

```python
from config import Config
from model_manager import get_model_manager
from text_generator import TextGenerator

# 초기화
config = Config()
manager = get_model_manager(config)
manager.load_model()
generator = TextGenerator(manager, config)

# 텍스트 생성
result = generator.generate(
    prompt="인공지능의 미래에 대해 설명해주세요",
    max_tokens=300,
    temperature=0.7
)

if result["success"]:
    print(f"응답: {result['response']}")
    print(f"생성 시간: {result['generation_time']:.2f}초")
    print(f"속도: {result['tokens_per_second']:.1f} 토큰/초")
else:
    print(f"오류: {result['error']}")
```

#### 대화 관리

```python
from text_generator import ConversationManager

# 대화 관리자 생성
conversation = ConversationManager(generator, max_history=10)

# 시스템 메시지 설정
conversation.set_system_message(
    "당신은 친근하고 도움이 되는 AI 어시스턴트입니다."
)

# 대화 시작
response1 = conversation.generate_response("안녕하세요!")
print(f"AI: {response1['response']}")

response2 = conversation.generate_response("오늘 날씨가 어떤가요?")
print(f"AI: {response2['response']}")

# 대화 히스토리 확인
summary = conversation.get_history_summary()
print(f"대화 메시지 수: {summary['total_messages']}")

# 대화 저장
filename = conversation.export_conversation()
print(f"대화 저장됨: {filename}")
```

#### 배치 처리

```python
# 여러 프롬프트 일괄 처리
prompts = [
    "파이썬의 장점을 설명해주세요",
    "머신러닝과 딥러닝의 차이점은 무엇인가요?",
    "클라우드 컴퓨팅의 미래는 어떨까요?"
]

results = generator.batch_generate(
    prompts, 
    max_tokens=200, 
    temperature=0.6
)

for i, result in enumerate(results):
    if result["success"]:
        print(f"질문 {i+1}: {prompts[i]}")
        print(f"답변: {result['response']}\n")
```

## 🖼️ 시각-언어 모델 사용법

### CLI를 통한 이미지 분석

#### 이미지 설명

```bash
# 로컬 이미지 파일
python cli_interface.py vision image.jpg --task describe

# 웹 이미지 URL
python cli_interface.py vision "https://example.com/image.jpg" --task describe

# 상세 설명 요청
python cli_interface.py vision photo.png --task describe \
  --max-tokens 400 --temperature 0.1
```

#### OCR (텍스트 추출)

```bash
# 문서 이미지에서 텍스트 추출
python cli_interface.py vision document.png --task ocr

# 명함이나 간판 텍스트 추출
python cli_interface.py vision business_card.jpg --task ocr \
  --output extracted_text.txt
```

#### 차트 및 표 분석

```bash
# 차트 분석
python cli_interface.py vision chart.png --task chart

# 표 데이터 추출
python cli_interface.py vision table.jpg --task table

# 문서를 마크다운으로 변환
python cli_interface.py vision document.png --task markdown \
  --output document.md
```

#### 시각적 질문답변

```bash
# 이미지에 대한 구체적 질문
python cli_interface.py vision photo.jpg --task qa \
  --prompt "이 사진에서 사람이 몇 명 보이나요?"

# 복잡한 분석 요청
python cli_interface.py vision scene.jpg --task qa \
  --prompt "이 장면의 분위기와 감정을 분석해주세요" \
  --max-tokens 300
```

### Python API를 통한 이미지 처리

#### 기본 이미지 분석

```python
from vision_generator import VisionGenerator

# 시각-언어 생성기 초기화
vision_gen = VisionGenerator(manager, config)

# 이미지 설명 생성
result = vision_gen.describe_image("path/to/image.jpg")
if result["success"]:
    print(f"이미지 설명: {result['response']}")

# OCR 텍스트 추출
ocr_result = vision_gen.extract_text("document.png")
if ocr_result["success"]:
    print(f"추출된 텍스트: {ocr_result['response']}")

# 차트 분석
chart_result = vision_gen.analyze_chart("sales_chart.png")
print(f"차트 분석: {chart_result['response']}")
```

#### 다양한 이미지 입력 방식

```python
from PIL import Image
import requests

# 1. 로컬 파일 경로
result1 = vision_gen.describe_image("./images/photo.jpg")

# 2. URL
result2 = vision_gen.describe_image("https://example.com/image.png")

# 3. PIL Image 객체
image = Image.open("photo.jpg")
result3 = vision_gen.describe_image(image)

# 4. 바이트 데이터
with open("image.jpg", "rb") as f:
    image_bytes = f.read()
result4 = vision_gen.describe_image(image_bytes)

# 5. Base64 데이터 URL
import base64
with open("image.jpg", "rb") as f:
    encoded = base64.b64encode(f.read()).decode()
    data_url = f"data:image/jpeg;base64,{encoded}"
result5 = vision_gen.describe_image(data_url)
```

#### 사용자 정의 프롬프트

```python
# 맞춤형 이미지 분석
custom_result = vision_gen.generate_with_image(
    image="product_photo.jpg",
    prompt="이 제품의 특징과 장단점을 마케팅 관점에서 분석해주세요",
    max_tokens=400,
    temperature=0.3
)

# 전문적인 분석
medical_result = vision_gen.generate_with_image(
    image="xray.jpg",
    prompt="이 의료 이미지에서 주목할 만한 특징을 설명해주세요 (참고용)",
    max_tokens=200,
    temperature=0.1
)
```

#### 배치 이미지 처리

```python
# 여러 이미지 일괄 처리
image_files = [
    "photo1.jpg",
    "photo2.jpg", 
    "photo3.jpg"
]

batch_results = vision_gen.batch_analyze_images(
    image_files,
    prompt="이 이미지들의 공통점과 차이점을 분석해주세요",
    max_tokens=250
)

for i, result in enumerate(batch_results):
    if result["success"]:
        print(f"이미지 {i+1} 분석: {result['response']}")
```

## 📄 문서 처리 사용법

### CLI를 통한 문서 처리

#### 단일 문서 처리

```bash
# 텍스트 추출
python cli_interface.py document scan.png --task extract

# 문서 요약
python cli_interface.py document report.jpg --task summarize

# 마크다운 변환
python cli_interface.py document page.png --task markdown \
  --output converted.md

# 표 데이터 추출
python cli_interface.py document table_image.png --task table
```

#### 다중 페이지 문서

```bash
# 폴더 내 모든 이미지 처리
python cli_interface.py document ./document_pages/ \
  --task extract --multi-page --output full_document.txt

# 여러 파일 지정
python cli_interface.py document "page1.png,page2.png,page3.png" \
  --task markdown --multi-page
```

### Python API를 통한 문서 처리

#### 문서 처리기 사용

```python
from vision_generator import DocumentProcessor

# 문서 처리기 초기화
doc_processor = DocumentProcessor(vision_gen)

# 단일 페이지 처리
result = doc_processor.process_document_page(
    "contract.png",
    task="extract"  # extract, summarize, markdown, table
)

if result["success"]:
    print(f"문서 내용: {result['response']}")
```

#### 다중 페이지 문서 처리

```python
import os

# 디렉토리에서 이미지 파일 찾기
def get_image_files(directory):
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    files = []
    for file in sorted(os.listdir(directory)):
        if file.lower().endswith(supported_formats):
            files.append(os.path.join(directory, file))
    return files

# 다중 페이지 처리
image_files = get_image_files("./scanned_document/")
result = doc_processor.process_multi_page_document(
    image_files,
    task="markdown"
)

print(f"처리된 페이지 수: {result['total_pages']}")
print(f"성공한 페이지 수: {result['successful_pages']}")
print(f"전체 내용:\n{result['combined_content']}")

# 결과를 파일로 저장
with open("processed_document.md", "w", encoding="utf-8") as f:
    f.write(result['combined_content'])
```

## 📦 배치 처리 사용법

### 텍스트 배치 처리

#### 텍스트 파일에서 프롬프트 읽기

```bash
# prompts.txt 파일 생성
echo "인공지능의 역사를 설명해주세요" > prompts.txt
echo "파이썬 프로그래밍의 장점은 무엇인가요?" >> prompts.txt
echo "클라우드 컴퓨팅이란 무엇인가요?" >> prompts.txt

# 배치 처리 실행
python cli_interface.py batch prompts.txt --output results.json
```

#### JSON 형식 배치 처리

```json
// batch_tasks.json
[
  {
    "prompt": "한국의 전통음식에 대해 설명해주세요",
    "max_tokens": 200,
    "temperature": 0.7
  },
  {
    "prompt": "K-POP의 세계적 인기 이유는 무엇인가요?",
    "max_tokens": 250,
    "temperature": 0.6
  },
  {
    "image": "chart.png",
    "prompt": "이 차트를 분석해주세요",
    "max_tokens": 300
  }
]
```

```bash
# JSON 배치 처리
python cli_interface.py batch batch_tasks.json --output detailed_results.json
```

### Python API를 통한 배치 처리

```python
import json
from datetime import datetime

# 배치 작업 정의
batch_tasks = [
    {
        "type": "text",
        "prompt": "머신러닝과 딥러닝의 차이점",
        "params": {"max_tokens": 200, "temperature": 0.5}
    },
    {
        "type": "vision",
        "image": "diagram.png",
        "prompt": "이 다이어그램을 설명해주세요",
        "params": {"max_tokens": 250}
    }
]

# 배치 처리 실행
results = []
for i, task in enumerate(batch_tasks):
    print(f"작업 {i+1}/{len(batch_tasks)} 처리 중...")
    
    if task["type"] == "text":
        result = generator.generate(
            task["prompt"],
            **task.get("params", {})
        )
    elif task["type"] == "vision":
        result = vision_gen.generate_with_image(
            task["image"],
            task["prompt"],
            **task.get("params", {})
        )
    
    results.append({
        "task_id": i,
        "task": task,
        "result": result,
        "timestamp": datetime.now().isoformat()
    })
    
    # 메모리 정리
    manager.clear_memory()

# 결과 저장
with open("batch_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"배치 처리 완료: {len(results)}개 작업")
```

## ⚙️ 고급 설정 및 최적화

### 성능 파라미터 조정

#### 생성 품질 vs 속도 균형

```python
# 고품질 모드 (느림)
high_quality_params = {
    "max_tokens": 400,
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1
}

# 빠른 응답 모드 (품질 타협)
fast_response_params = {
    "max_tokens": 150,
    "temperature": 0.1,
    "do_sample": False
}

# 창의적 모드 (다양한 응답)
creative_params = {
    "max_tokens": 300,
    "temperature": 0.9,
    "top_p": 0.95,
    "do_sample": True
}
```

#### 메모리 최적화 설정

```python
# config.yaml 수정
generation:
  max_tokens: 200        # 토큰 수 제한
  use_cache: false       # 캐시 비활성화
  
hardware:
  max_memory_usage: 0.85 # VRAM 85%까지만 사용
  
quantization:
  load_in_4bit: true     # 4비트 양자화 활성화
```

### 시스템 메시지 활용

#### 역할 기반 시스템 메시지

```python
# 전문가 역할
expert_messages = {
    "teacher": "당신은 친근하고 이해하기 쉽게 설명하는 교사입니다.",
    "translator": "당신은 정확하고 자연스러운 번역을 제공하는 전문 번역가입니다.",
    "writer": "당신은 창의적이고 매력적인 글을 쓰는 작가입니다.",
    "analyst": "당신은 데이터를 정확하게 분석하고 인사이트를 제공하는 분석가입니다."
}

# 대화 스타일 설정
conversation.set_system_message(expert_messages["teacher"])
```

#### 출력 형식 지정

```python
# 구조화된 출력 요청
structured_prompt = """
다음 형식으로 답변해주세요:

## 요약
- 핵심 내용 3가지

## 상세 설명
- 각 항목에 대한 자세한 설명

## 결론
- 최종 요약 및 제언

질문: {user_question}
"""

result = generator.generate(
    structured_prompt.format(user_question="인공지능의 미래"),
    max_tokens=400
)
```

### 오류 처리 및 재시도

```python
import time

def robust_generate(generator, prompt, max_retries=3):
    """오류 처리를 포함한 안정적인 생성 함수"""
    
    for attempt in range(max_retries):
        try:
            result = generator.generate(prompt)
            
            if result["success"]:
                return result
            else:
                print(f"시도 {attempt + 1} 실패: {result['error']}")
                
        except Exception as e:
            print(f"시도 {attempt + 1} 예외 발생: {e}")
            
        # 재시도 전 메모리 정리 및 대기
        generator.model_manager.clear_memory()
        time.sleep(2)
    
    return {"success": False, "error": "모든 재시도 실패"}

# 사용 예시
result = robust_generate(generator, "복잡한 질문")
```

## 📊 모니터링 및 로깅

### 실시간 성능 모니터링

```python
import time
import psutil

class PerformanceMonitor:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.start_time = None
        
    def start_monitoring(self):
        self.start_time = time.time()
        print("🔍 성능 모니터링 시작")
        
    def log_performance(self, operation_name):
        if self.start_time:
            elapsed = time.time() - self.start_time
            
            # GPU 메모리 사용량
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_usage = (gpu_memory / gpu_total) * 100
            else:
                gpu_memory = gpu_usage = 0
                
            # CPU 및 RAM 사용량
            cpu_usage = psutil.cpu_percent()
            ram = psutil.virtual_memory()
            
            print(f"📊 {operation_name} 성능:")
            print(f"   실행 시간: {elapsed:.2f}초")
            print(f"   GPU 메모리: {gpu_memory:.1f}GB ({gpu_usage:.1f}%)")
            print(f"   CPU 사용률: {cpu_usage:.1f}%")
            print(f"   RAM 사용률: {ram.percent:.1f}%")

# 사용 예시
monitor = PerformanceMonitor(manager)
monitor.start_monitoring()

result = generator.generate("성능 테스트 질문")
monitor.log_performance("텍스트 생성")
```

### 로그 파일 관리

```python
import logging
from datetime import datetime

# 로깅 설정
def setup_logging():
    """상세 로깅 설정"""
    
    # 로그 디렉토리 생성
    os.makedirs("logs", exist_ok=True)
    
    # 로그 포맷 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 파일 핸들러 (일별 로그 파일)
    today = datetime.now().strftime("%Y%m%d")
    file_handler = logging.FileHandler(f"logs/bllossom_{today}.log")
    file_handler.setFormatter(formatter)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 루트 로거 설정
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 사용 예시
logger = setup_logging()

logger.info("프로그램 시작")
logger.info(f"모델 로딩: {config.model.name}")

try:
    result = generator.generate("테스트 프롬프트")
    logger.info(f"생성 성공: {result['tokens_per_second']:.1f} t/s")
except Exception as e:
    logger.error(f"생성 실패: {e}")
```

## 🎨 실제 사용 사례

### 1. 교육 콘텐츠 생성

```python
def create_educational_content(topic, grade_level="중학생"):
    """교육용 콘텐츠 생성"""
    
    prompt = f"""
{grade_level} 수준에 맞게 '{topic}'에 대한 학습 자료를 만들어주세요.

다음 형식으로 작성해주세요:
1. 개념 설명 (쉬운 언어로)
2. 실생활 예시 3가지
3. 기억하기 쉬운 방법
4. 간단한 퀴즈 문제 2개

길이: 300-400단어
"""
    
    result = generator.generate(prompt, max_tokens=500, temperature=0.6)
    return result['response'] if result['success'] else None

# 사용 예시
content = create_educational_content("광합성", "초등학생")
print(content)
```

### 2. 문서 자동 요약

```python
def summarize_document(document_text, summary_type="executive"):
    """문서 자동 요약"""
    
    summary_prompts = {
        "executive": "다음 문서를 경영진을 위한 요약으로 만들어주세요 (3-5줄):",
        "detailed": "다음 문서의 상세한 요약을 만들어주세요 (10-15줄):",
        "bullet": "다음 문서의 핵심 내용을 불릿 포인트로 정리해주세요:"
    }
    
    prompt = f"{summary_prompts[summary_type]}\n\n{document_text}"
    
    result = generator.generate(
        prompt, 
        max_tokens=300, 
        temperature=0.3
    )
    
    return result['response'] if result['success'] else None

# 사용 예시
with open("report.txt", "r", encoding="utf-8") as f:
    document = f.read()

summary = summarize_document(document, "executive")
print(f"요약:\n{summary}")
```

### 3. 다국어 번역

```python
def translate_text(text, target_language="영어"):
    """텍스트 번역"""
    
    prompt = f"""
다음 텍스트를 {target_language}로 자연스럽게 번역해주세요:

원문: {text}

번역할 때 다음을 고려해주세요:
- 문맥과 뉘앙스 유지
- 자연스러운 표현 사용
- 문화적 차이 고려

번역:
"""
    
    result = generator.generate(prompt, max_tokens=200, temperature=0.3)
    return result['response'] if result['success'] else None

# 사용 예시
korean_text = "오늘 날씨가 정말 좋네요. 산책하기 딱 좋은 날씨입니다."
english_translation = translate_text(korean_text, "영어")
print(f"번역: {english_translation}")
```

### 4. 창작 지원 도구

```python
def creative_writing_assistant(genre, theme, length="short"):
    """창작 지원 도구"""
    
    length_guide = {
        "short": "200-300단어의 짧은",
        "medium": "500-700단어의 중간 길이",
        "long": "1000단어 이상의 긴"
    }
    
    prompt = f"""
다음 조건으로 {length_guide[length]} {genre} 작품을 써주세요:

장르: {genre}
주제: {theme}

작품 요구사항:
- 흥미진진하고 독창적인 내용
- 생생한 묘사와 대화
- 명확한 시작, 전개, 결말
- 읽는 재미가 있는 문체

작품:
"""
    
    max_tokens = {"short": 400, "medium": 800, "long": 1200}[length]
    
    result = generator.generate(
        prompt, 
        max_tokens=max_tokens, 
        temperature=0.8
    )
    
    return result['response'] if result['success'] else None

# 사용 예시
story = creative_writing_assistant("SF 소설", "시간 여행", "medium")
print(f"창작 소설:\n{story}")
```

---

## 💡 사용 팁

### 효과적인 프롬프트 작성법

1. **구체적이고 명확하게**: 모호한 질문보다는 구체적인 요청
2. **예시 제공**: 원하는 형식이나 스타일의 예시 포함
3. **단계별 요청**: 복잡한 작업은 단계별로 나누어 요청
4. **제약 조건 명시**: 길이, 형식, 톤 등의 제약 조건 명시

### 성능 최적화 요령

1. **적절한 토큰 수**: 필요한 만큼만 생성하여 속도 향상
2. **온도 조절**: 일관된 결과가 필요하면 낮은 온도 사용
3. **메모리 관리**: 큰 작업 후에는 메모리 정리
4. **배치 크기**: 한 번에 너무 많은 작업 피하기

### 문제 발생 시 대처법

1. **메모리 부족**: 더 작은 배치 크기나 토큰 수 사용
2. **느린 응답**: 네트워크 상태 확인, 캐시 정리
3. **품질 저하**: 프롬프트 개선, 파라미터 조정
4. **오류 발생**: 로그 확인, 재시작 시도

이 가이드를 통해 Korean Bllossom AICA-5B 모델을 효과적으로 활용하시기 바랍니다!