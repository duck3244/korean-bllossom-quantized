# 🏛️ 아키텍처 (Architecture)

> Korean Bllossom AICA-5B 양자화 프로젝트의 시스템 아키텍처 문서

---

## 1. 개요 (Overview)

본 프로젝트는 **Bllossom/llama-3.2-Korean-Bllossom-AICA-5B** (Mllama 계열 한국어 비전-언어 모델)을 **RTX 4060 8GB** 환경에서 실행하기 위해 4-bit NF4 양자화를 적용한 챗봇 시스템입니다.

크게 두 개의 런타임으로 구성됩니다.

| 구성요소 | 기술 스택 | 역할 |
|---------|----------|------|
| **Backend** | Python 3.8+, FastAPI, PyTorch, Transformers, bitsandbytes | 모델 로딩 / 추론 / HTTP 스트리밍 / CLI |
| **Frontend** | React 18, TypeScript, Vite, TailwindCSS | 채팅 UI / 토큰 스트림 렌더링 |

---

## 2. 상위 수준 아키텍처 (High-Level Architecture)

```
┌──────────────────────────────────────────────────────────────────────┐
│                          사용자 (Browser / Shell)                    │
└─────────────────┬──────────────────────────────┬─────────────────────┘
                  │                              │
                  │ HTTP (fetch + Stream)        │ stdin/stdout
                  ▼                              ▼
┌──────────────────────────────┐   ┌───────────────────────────────────┐
│  Frontend (Vite Dev Server)  │   │   CLI Interface                   │
│  - App.tsx                   │   │   - cli_interface.py              │
│  - ChatInput / ChatMessage   │   │   - main.py                       │
│  - api/chat.ts (streamChat)  │   └───────────────┬───────────────────┘
└──────────────┬───────────────┘                   │
               │ POST /api/chat (multipart)        │
               │ POST /api/reset                   │
               │ GET  /api/health                  │
               ▼                                   │
┌──────────────────────────────────────────────────▼───────────────────┐
│                        FastAPI Application (api.py)                  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ AppState  (config / manager / lock / history)                  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────┐    ┌──────────────────────────────┐   │
│  │  /api/chat  (Streaming)   │    │   _stream_generation()       │   │
│  │  - multipart upload       │ ─► │   - TextIteratorStreamer     │   │
│  │  - serialize via asyncio  │    │   - background Thread        │   │
│  └───────────────────────────┘    └────────────────┬─────────────┘   │
└─────────────────────────────────────────────────────┼────────────────┘
                                                      │
                                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          Core Layer (backend/core)                   │
│  ┌───────────────┐ ┌──────────────────┐ ┌──────────────────────────┐ │
│  │   Config      │ │  ModelManager    │ │  TextGenerator           │ │
│  │ (config.py)   │ │ (싱글톤, GPU)    │ │  VisionGenerator         │ │
│  │  YAML 로드    │ │  4-bit NF4 양자화│ │  ConversationManager     │ │
│  └───────────────┘ └─────────┬────────┘ │  DocumentProcessor       │ │
│                              │           └──────────────────────────┘ │
└──────────────────────────────┼───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│         HuggingFace Transformers  +  bitsandbytes  +  PyTorch CUDA   │
│                  Mllama-3.2-Korean-Bllossom-AICA-5B                  │
│                       (NF4 / bfloat16 / device_map=auto)             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. 계층 구조 (Layered Architecture)

본 시스템은 **4-계층 아키텍처**를 따릅니다.

### 3.1 Presentation Layer
사용자 입력을 받고 결과를 시각화합니다.

| 모듈 | 책임 |
|------|------|
| `frontend/src/App.tsx` | 채팅 상태 관리, 메시지 리스트 렌더링, 스크롤 |
| `frontend/src/components/ChatInput.tsx` | 텍스트/이미지 입력, IME-safe Enter 처리 |
| `frontend/src/components/ChatMessage.tsx` | 사용자/어시스턴트 말풍선, 스트리밍 커서 |
| `backend/interfaces/cli_interface.py` | 명령줄 채팅 / 이미지 분석 / 배치 처리 |

### 3.2 API / Service Layer
HTTP 라우팅과 요청 직렬화를 담당합니다.

| 모듈 | 책임 |
|------|------|
| `backend/api.py` | FastAPI 앱, CORS, lifespan, 스트리밍 응답 |
| `frontend/src/api/chat.ts` | `streamChat()` / `resetChat()` — fetch + ReadableStream |

핵심 엔드포인트:

| Method | Path | 설명 |
|--------|------|------|
| `GET`  | `/api/health` | 모델 로드 상태 / 대화 턴 수 |
| `POST` | `/api/reset`  | 서버 대화 히스토리 초기화 |
| `POST` | `/api/chat`   | `multipart/form-data` (prompt, image?) → `text/plain` 청크 스트림 |

### 3.3 Domain / Core Layer
모델 추론과 도메인 로직을 캡슐화합니다.

| 클래스 | 책임 |
|--------|------|
| `Config` | YAML 기반 설정 로드/저장, 5개 sub-config(`Model/Quantization/Generation/Hardware/Path`) |
| `ModelManager` | 모델·프로세서 로딩, VRAM 모니터링, 헬스 체크, 추론 최적화 (싱글톤) |
| `TextGenerator` | 텍스트 생성, 배치 생성, 응답 파싱 |
| `ConversationManager` | 대화 히스토리, 시스템 메시지, export |
| `VisionGenerator` | 이미지+텍스트 생성, OCR/차트/표/마크다운 변환 |
| `DocumentProcessor` | 다중 페이지 문서 처리 |

### 3.4 Infrastructure Layer
외부 라이브러리와 하드웨어를 추상화합니다.

- **HuggingFace Transformers** — `MllamaForConditionalGeneration`, `MllamaProcessor`, `TextIteratorStreamer`
- **bitsandbytes** — `BitsAndBytesConfig` (NF4 4-bit, double quant)
- **PyTorch / CUDA** — `device_map="auto"`, `bfloat16` compute dtype
- **PIL / requests** — 이미지 입력 정규화 (file / URL / base64 / bytes)

---

## 4. 주요 구성요소 (Key Components)

### 4.1 `ModelManager` (싱글톤)
- 프로세스 전역에서 **단 하나의 모델 인스턴스**만 유지합니다 (`_model_manager_instance`).
- 4-bit NF4 양자화 + `bfloat16` 계산 dtype으로 VRAM 75% 절감.
- `clear_memory()`로 매 요청 후 GPU 캐시를 회수합니다.

### 4.2 `AppState` (FastAPI 단일 사용자 MVP)
```python
@dataclass
class AppState:
    config:  Optional[Config]
    manager: Optional[ModelManager]
    lock:    asyncio.Lock          # GPU 직렬화
    history: List[ChatTurn]        # 텍스트 전용 히스토리
```
- **단일 GPU + 단일 모델 인스턴스**라는 제약 때문에 `asyncio.Lock`으로 동시 요청을 직렬화합니다.
- 이미 처리 중이면 즉시 **HTTP 429**를 반환합니다.
- **비전 요청은 히스토리에 누적하지 않습니다** — multimodal 히스토리는 깨지기 쉬워 MVP 범위 밖.

### 4.3 스트리밍 파이프라인
1. 클라이언트가 `POST /api/chat`을 multipart로 호출.
2. FastAPI가 `state.lock`을 획득.
3. `TextIteratorStreamer` 인스턴스 생성 → `model.generate()`를 **별도 데몬 스레드**에서 실행.
4. 메인 코루틴은 `streamer`로부터 token을 `next()` → `StreamingResponse`로 `utf-8` 청크 yield.
5. 프론트엔드의 `ReadableStream.getReader()`가 청크를 받아 `setMessages()`로 누적.

### 4.4 한국어 IME 안전성
- 한글 조합 중 Enter는 전송하지 않도록 `e.nativeEvent.isComposing`을 검사합니다 (`ChatInput.tsx`).
- "한국어 챗 #1 footgun"을 사전 차단합니다.

### 4.5 비전 경로의 특수 처리
- Mllama 비전 분기는 `RepetitionPenaltyLogitsProcessor`와 충돌(scatter index OOB)하므로,
  이미지 입력 시에는 `top_p / top_k / repetition_penalty`를 **적용하지 않습니다**.
- 또한 비전 경로는 `use_cache=False`로 VRAM을 절약합니다.

---

## 5. 데이터 흐름 (Data Flow)

### 5.1 텍스트 채팅 흐름
```
사용자 입력
   │
   ▼
ChatInput.submit()  ─► App.send()  ─► streamChat() [api/chat.ts]
                                          │
                                          │ POST /api/chat (FormData)
                                          ▼
                                  FastAPI /api/chat
                                          │ acquire lock
                                          ▼
                            _build_text_messages(history, prompt)
                                          │
                                          ▼
                          processor.apply_chat_template(...)
                                          │
                                          ▼
                          processor(images=None, text=...)
                                          │
                                          ▼
                  model.generate(**kwargs, streamer=TextIteratorStreamer)
                                          │ (background Thread)
                                          ▼
                          async for token in streamer:
                              yield token.encode("utf-8")
                                          │
                                          ▼
                              StreamingResponse (chunked)
                                          │
                                          ▼
                      ReadableStream.getReader() in browser
                                          │
                                          ▼
                          onToken(delta) → setMessages(...)
```

### 5.2 비전 요청 흐름
```
이미지 + 프롬프트
   │
   ▼
multipart/form-data
   │
   ▼
PIL.Image.open() → RGB 변환 → _resize_if_needed(1024px)
   │
   ▼
_build_vision_messages(prompt)  → [{role:user, content:[{image},{text}]}]
   │
   ▼
processor(images=pil_image, text=...)
   │
   ▼
model.generate(use_cache=False, [샘플링 보정 없음])
   │
   ▼
TextIteratorStreamer → HTTP chunked stream
   │
   ▼
[히스토리에 저장하지 않음]
```

---

## 6. 동시성·신뢰성 (Concurrency & Reliability)

| 항목 | 결정 | 이유 |
|------|------|------|
| 동시 요청 수 | **1** (서버 전역 락) | 단일 GPU·단일 모델 인스턴스 — race 시 KV cache 손상 위험 |
| 백그라운드 스레드 | 데몬 스레드 | 프로세스 종료 시 자동 정리 |
| 타임아웃 | streamer `timeout=60s` | 모델 hang으로부터 보호 |
| OOM 처리 | `clear_memory()` 후 에러 반환 | 다음 요청을 위해 컨텍스트 복구 |
| 라이프사이클 | FastAPI `lifespan` | 시작 시 모델 로드 / 종료 시 unload |
| GPU 컨텍스트 오염 | `clear_memory()` 예외를 swallow | 원래 예외를 마스킹하지 않기 위함 |

---

## 7. 설정 관리 (Configuration)

설정은 **YAML → dataclass** 매핑 방식으로 로드됩니다.

```
backend/config.yaml
       │
       ▼
Config.load_from_yaml()
       │
       ├── ModelConfig         (name, dtype, device_map)
       ├── QuantizationConfig  (4bit, NF4, double_quant)
       ├── GenerationConfig    (max_tokens, temperature, top_p/k, rep_penalty)
       ├── HardwareConfig      (target_gpu, vram, cpu_offload)
       └── PathConfig          (cache_dir, log_dir, output_dir)
```

`cache_dir=None`이면 HuggingFace 기본 캐시(`~/.cache/huggingface/hub`)를 사용합니다.

---

## 8. 배포 토폴로지 (Deployment Topology)

### 8.1 개발 환경
```
┌──────────────────────────────────────────┐
│  localhost (개발자 머신)                 │
│                                          │
│  ┌─────────────────┐   ┌──────────────┐  │
│  │ Vite Dev Server │   │  uvicorn     │  │
│  │ :5173 (HMR)     │ ◄►│  :8000       │  │
│  └─────────────────┘   └──────┬───────┘  │
│                               │          │
│                        CUDA  ▼           │
│                    ┌────────────────┐    │
│                    │   RTX 4060 8GB │    │
│                    └────────────────┘    │
└──────────────────────────────────────────┘
```
- CORS 허용 origin: `http://127.0.0.1:5173`, `http://localhost:5173`
- 모델 스킵 모드: `MODEL_SKIP_LOAD=1 uvicorn api:app` — GPU 없이 UI 개발 가능

### 8.2 단일 사용자 MVP 제약
- 멀티 테넌트 / 멀티 GPU / 큐잉 / 인증은 **범위 밖**.
- 트래픽이 둘 이상일 가능성이 보이면 vLLM/TGI 같은 본격 서빙 스택으로 교체해야 합니다.

---

## 9. 디렉토리 구조 (Directory Layout)

```
korean-bllossom-quantized/
├── backend/
│   ├── api.py                       # FastAPI 진입점
│   ├── main.py                      # CLI/대화형 메뉴 진입점
│   ├── config.yaml                  # 런타임 설정
│   ├── core/
│   │   ├── config.py                # Config + dataclass 묶음
│   │   ├── model_manager.py         # 싱글톤 모델 관리자
│   │   ├── text_generator.py        # TextGenerator / ConversationManager
│   │   └── vision_generator.py      # VisionGenerator / DocumentProcessor
│   ├── interfaces/
│   │   └── cli_interface.py         # CLI 진입점
│   ├── requirements.txt
│   └── scripts/setup.sh
├── frontend/
│   ├── index.html
│   ├── src/
│   │   ├── App.tsx                  # 채팅 컨테이너
│   │   ├── main.tsx                 # React 부트스트랩
│   │   ├── components/
│   │   │   ├── ChatInput.tsx        # 입력 + 이미지 첨부
│   │   │   └── ChatMessage.tsx      # 말풍선 + 스트리밍 커서
│   │   └── api/chat.ts              # fetch + ReadableStream
│   ├── vite.config.ts
│   └── tailwind.config.js
└── docs/
    ├── ARCHITECTURE.md              # 본 문서
    └── UML.md                       # 클래스/시퀀스/컴포넌트 다이어그램
```

---

## 10. 비기능 요구사항 (Non-Functional Requirements)

| 항목 | 목표 | 달성 수단 |
|------|------|----------|
| VRAM 사용량 | < 7 GB | NF4 4-bit + double quant + `bfloat16` |
| 생성 속도 | 15-25 tok/s | KV cache 활성화 (텍스트 경로) |
| TTFB (Time-to-first-token) | < 3 s | streaming + lock의 빠른 획득 |
| 한국어 IME 안전성 | 100% | `isComposing` 검사 |
| 가용성 | 단일 인스턴스 best-effort | lifespan 종료 시 unload |
| 관찰가능성 | 콘솔 로그 + VRAM 통계 | `logging` + `psutil` + `torch.cuda` |

---

## 11. 향후 확장 포인트 (Extension Points)

- **인증 / 멀티 유저** — `AppState.history` → 세션 키 기반 dict
- **백엔드 큐잉** — 단일 락 → `asyncio.Queue` + 워커
- **모델 백엔드 교체** — `ModelManager` 추상화 인터페이스 정의 후 vLLM 어댑터
- **이미지 히스토리** — Mllama 멀티턴 비전 지원이 안정화되면 vision 턴도 history에 누적
- **스트리밍 SSE 전환** — 현재 `text/plain` 청크 → `text/event-stream`으로 메타데이터 전달
