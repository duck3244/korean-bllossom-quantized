# 📐 UML 다이어그램 (UML Diagrams)

> Korean Bllossom AICA-5B 프로젝트의 UML 다이어그램 모음 (Mermaid 기반)

GitHub, VS Code, JetBrains IDE는 Mermaid를 기본 렌더링하므로 본 문서를 그대로 열면 도식이 보입니다.

---

## 목차
1. [클래스 다이어그램 (Backend Core)](#1-클래스-다이어그램-backend-core)
2. [클래스 다이어그램 (FastAPI Layer)](#2-클래스-다이어그램-fastapi-layer)
3. [컴포넌트 다이어그램 (Frontend)](#3-컴포넌트-다이어그램-frontend)
4. [시퀀스 다이어그램 — 텍스트 채팅](#4-시퀀스-다이어그램--텍스트-채팅)
5. [시퀀스 다이어그램 — 이미지 + 텍스트](#5-시퀀스-다이어그램--이미지--텍스트)
6. [시퀀스 다이어그램 — 대화 초기화](#6-시퀀스-다이어그램--대화-초기화)
7. [상태 다이어그램 — ModelManager](#7-상태-다이어그램--modelmanager)
8. [활동 다이어그램 — 스트리밍 생성](#8-활동-다이어그램--스트리밍-생성)
9. [배포 다이어그램](#9-배포-다이어그램)
10. [패키지 다이어그램](#10-패키지-다이어그램)

---

## 1. 클래스 다이어그램 (Backend Core)

`backend/core` 패키지의 도메인 모델입니다.

```mermaid
classDiagram
    direction LR

    class Config {
        +ModelConfig model
        +QuantizationConfig quantization
        +GenerationConfig generation
        +HardwareConfig hardware
        +PathConfig paths
        +__init__(config_file)
        +load_from_yaml(path)
        +save_to_yaml(path)
        +print_config()
        -_create_directories()
    }

    class ModelConfig {
        +str name
        +bool trust_remote_code
        +str torch_dtype
        +bool low_cpu_mem_usage
        +str device_map
    }

    class QuantizationConfig {
        +bool load_in_4bit
        +str bnb_4bit_quant_type
        +str bnb_4bit_compute_dtype
        +bool bnb_4bit_use_double_quant
    }

    class GenerationConfig {
        +int max_tokens
        +float temperature
        +bool do_sample
        +float top_p
        +int top_k
        +float repetition_penalty
    }

    class HardwareConfig {
        +str target_gpu
        +int target_vram_gb
        +bool use_cpu_offload
        +float max_memory_usage
    }

    class PathConfig {
        +str|None cache_dir
        +str log_dir
        +str output_dir
        +str config_file
    }

    class ModelManager {
        <<singleton>>
        +Config config
        +MllamaForConditionalGeneration model
        +MllamaProcessor processor
        +str device
        +bool is_loaded
        +load_model() bool
        +unload_model()
        +clear_memory()
        +get_model_info() dict
        +health_check() bool
        +optimize_for_inference()
        -_setup_quantization_config()
        -_check_system_requirements()
        -_print_memory_usage()
    }

    class TextGenerator {
        +ModelManager model_manager
        +Config config
        +generate(prompt, **kwargs) dict
        +chat_generate(messages, **kwargs) dict
        +batch_generate(prompts, **kwargs) list
        +stream_generate(prompt, **kwargs)
        -_extract_response(full, input) str
    }

    class ConversationManager {
        +TextGenerator text_generator
        +int max_history
        +list conversation_history
        +str|None system_message
        +set_system_message(msg)
        +add_message(role, content)
        +generate_response(user_input) dict
        +clear_history()
        +get_history_summary() dict
        +export_conversation(filename) str
    }

    class VisionGenerator {
        +ModelManager model_manager
        +Config config
        +list supported_formats
        +generate_with_image(image, prompt) dict
        +describe_image(image) dict
        +extract_text(image) dict
        +analyze_chart(image) dict
        +analyze_table(image) dict
        +convert_to_markdown(image) dict
        +answer_visual_question(image, q) dict
        +batch_analyze_images(images, prompt) list
        -_load_image(image) Image
        -_preprocess_image(image) Image
        -_extract_response(full, input) str
    }

    class DocumentProcessor {
        +VisionGenerator vision_generator
        +process_document_page(image, task) dict
        +process_multi_page_document(images, task) dict
    }

    Config *-- ModelConfig
    Config *-- QuantizationConfig
    Config *-- GenerationConfig
    Config *-- HardwareConfig
    Config *-- PathConfig

    ModelManager --> Config : uses
    TextGenerator --> ModelManager : uses
    TextGenerator --> Config : uses
    VisionGenerator --> ModelManager : uses
    VisionGenerator --> Config : uses
    ConversationManager --> TextGenerator : uses
    DocumentProcessor --> VisionGenerator : uses
```

---

## 2. 클래스 다이어그램 (FastAPI Layer)

`backend/api.py`의 HTTP 계층 데이터 모델입니다.

```mermaid
classDiagram
    direction TB

    class ChatTurn {
        <<dataclass>>
        +str role
        +str text
    }

    class AppState {
        <<dataclass>>
        +Optional~Config~ config
        +Optional~ModelManager~ manager
        +asyncio.Lock lock
        +List~ChatTurn~ history
    }

    class FastAPI_App {
        <<FastAPI>>
        +AppState state
        +lifespan(app)
        +health() dict
        +reset() dict
        +chat(prompt, image, max_new_tokens, temperature) StreamingResponse
    }

    class StreamHelpers {
        <<module functions>>
        +_build_text_messages(history, prompt) list
        +_build_vision_messages(prompt) list
        +_resize_if_needed(image, max_side) Image
        +_stream_generation(prompt, image, ...) AsyncIterator
    }

    class TextIteratorStreamer {
        <<transformers>>
        +__iter__()
        +__next__() str
    }

    AppState *-- ChatTurn : history
    AppState --> ModelManager : manager
    AppState --> Config : config

    FastAPI_App --> AppState
    FastAPI_App --> StreamHelpers : uses
    StreamHelpers --> TextIteratorStreamer : creates
    StreamHelpers --> ModelManager : invokes generate()
```

---

## 3. 컴포넌트 다이어그램 (Frontend)

React 트리와 모듈 의존성입니다.

```mermaid
classDiagram
    direction LR

    class App {
        <<React.FC>>
        -Message[] messages
        -boolean busy
        -AbortController abortRef
        +send(text, image)
        +cancel()
        +newChat()
    }

    class ChatInput {
        <<React.FC>>
        -string text
        -File|null image
        -string|null previewUrl
        +submit()
        +onKeyDown(e)
    }

    class ChatMessage {
        <<React.FC>>
        +Message message
        +boolean streaming
    }

    class Message {
        <<type>>
        +string id
        +Role role
        +string content
        +string|null imageUrl
    }

    class ChatApi {
        <<module>>
        +streamChat(req, handlers)
        +resetChat()
    }

    class StreamHandlers {
        <<interface>>
        +onToken(delta)
        +onDone()
        +onError(err)
        +AbortSignal signal
    }

    App --> ChatInput : renders
    App --> ChatMessage : renders[]
    App --> ChatApi : invokes
    ChatInput ..> Message : produces
    ChatMessage ..> Message : displays
    ChatApi ..> StreamHandlers : accepts
```

---

## 4. 시퀀스 다이어그램 — 텍스트 채팅

사용자가 텍스트를 보내고 토큰 스트림을 받는 정상 경로입니다.

```mermaid
sequenceDiagram
    autonumber
    actor User as 사용자
    participant UI as ChatInput
    participant App as App.tsx
    participant Api as api/chat.ts
    participant FastAPI as FastAPI /api/chat
    participant State as AppState
    participant Gen as _stream_generation
    participant Streamer as TextIteratorStreamer
    participant Model as model.generate (Thread)

    User->>UI: 메시지 입력 + Enter
    UI->>App: onSubmit(text, null)
    App->>App: setMessages([+user, +empty assistant])
    App->>Api: streamChat({prompt}, {onToken,...})
    Api->>FastAPI: POST /api/chat (FormData)
    FastAPI->>State: lock.acquire()
    alt 이미 처리 중
        FastAPI-->>Api: 429 Too Many Requests
    else 사용 가능
        FastAPI->>Gen: _stream_generation(prompt, None, ...)
        Gen->>Gen: _build_text_messages(history, prompt)
        Gen->>Gen: processor(images=None, text=...)
        Gen->>Streamer: new TextIteratorStreamer(timeout=60)
        Gen-->>Model: Thread.start(model.generate(streamer=...))
        loop 토큰마다
            Model->>Streamer: put(token)
            Gen->>Streamer: next()
            Gen-->>FastAPI: yield token
            FastAPI-->>Api: chunked utf-8
            Api->>App: onToken(delta)
            App->>App: setMessages(append delta)
        end
        Model-->>Gen: generate 종료
        Gen->>State: history.append(user, assistant)
        Gen->>State: lock.release()
        FastAPI->>State: manager.clear_memory()
    end
```

---

## 5. 시퀀스 다이어그램 — 이미지 + 텍스트

비전 경로는 히스토리를 누적하지 않고 샘플링 옵션도 보수적으로 적용합니다.

```mermaid
sequenceDiagram
    autonumber
    actor User as 사용자
    participant UI as ChatInput
    participant App as App.tsx
    participant Api as api/chat.ts
    participant FastAPI as FastAPI /api/chat
    participant Image as PIL.Image
    participant Gen as _stream_generation
    participant Model as model.generate (Thread)

    User->>UI: 이미지 첨부 + 텍스트 입력
    UI->>App: onSubmit(text, File)
    App->>Api: streamChat({prompt, image}, handlers)
    Api->>FastAPI: POST /api/chat (multipart: prompt + image)
    FastAPI->>Image: Image.open(bytes)
    Image-->>FastAPI: PIL.Image (RGB)
    FastAPI->>FastAPI: _resize_if_needed(<=1024px)
    FastAPI->>Gen: _stream_generation(prompt, image, ...)
    Gen->>Gen: _build_vision_messages(prompt)
    Gen->>Gen: processor(images=image, text=...)
    Note over Gen: do_sample만 적용<br/>top_p/top_k/rep_penalty 미적용<br/>use_cache=False
    Gen-->>Model: Thread.start(model.generate(...))
    loop 토큰마다
        Model->>Gen: streamer.next()
        Gen-->>Api: yield token (utf-8)
        Api->>App: onToken(delta)
    end
    Note over Gen: 비전 턴은 history에 저장하지 않음
    FastAPI->>FastAPI: lock.release() + clear_memory()
```

---

## 6. 시퀀스 다이어그램 — 대화 초기화

```mermaid
sequenceDiagram
    autonumber
    actor User as 사용자
    participant App as App.tsx
    participant Api as api/chat.ts
    participant FastAPI as FastAPI /api/reset
    participant State as AppState

    User->>App: "새 대화" 클릭
    alt busy 상태
        App->>App: cancel() → abortRef.abort()
    end
    App->>Api: resetChat()
    Api->>FastAPI: POST /api/reset
    FastAPI->>State: history.clear()
    State-->>FastAPI: ok
    FastAPI-->>Api: 200 {ok:true}
    Api-->>App: resolve
    App->>App: setMessages([])
```

---

## 7. 상태 다이어그램 — ModelManager

```mermaid
stateDiagram-v2
    [*] --> Uninitialized
    Uninitialized --> CheckingRequirements : __init__()

    CheckingRequirements --> Unloaded : 요구사항 OK
    CheckingRequirements --> [*] : 치명적 환경 오류

    Unloaded --> Loading : load_model()
    Loading --> Loaded : 성공
    Loading --> Unloaded : OOM / 예외 (clear_memory)

    Loaded --> Optimized : optimize_for_inference()
    Optimized --> Loaded : (no-op, 항상 로드 상태)

    Loaded --> HealthChecking : health_check()
    HealthChecking --> Loaded : 통과
    HealthChecking --> Degraded : 실패

    Loaded --> Unloaded : unload_model()
    Degraded --> Unloaded : unload_model()
    Unloaded --> [*] : 프로세스 종료
```

---

## 8. 활동 다이어그램 — 스트리밍 생성

```mermaid
flowchart TD
    Start([POST /api/chat 도착]) --> Check{model_loaded?}
    Check -- no --> R503[503 Model not loaded]
    Check -- yes --> Empty{prompt 비어있나?}
    Empty -- yes --> R400[400 Empty prompt]
    Empty -- no --> HasImg{이미지 첨부?}

    HasImg -- yes --> ImgOpen[PIL.Image.open]
    ImgOpen --> ImgValid{유효한 이미지?}
    ImgValid -- no --> R400b[400 Bad image]
    ImgValid -- yes --> Resize[필요시 리사이즈 1024px]
    Resize --> Lock

    HasImg -- no --> Lock{lock 잠겨있나?}
    Lock -- yes --> R429[429 Too Many Requests]
    Lock -- no --> Acquire[lock.acquire]

    Acquire --> Build{이미지 있나?}
    Build -- 텍스트 --> BuildText[_build_text_messages history+prompt]
    Build -- 비전 --> BuildVision[_build_vision_messages prompt]

    BuildText --> Tokenize[processor.apply_chat_template + processor]
    BuildVision --> Tokenize
    Tokenize --> Streamer[TextIteratorStreamer 생성]
    Streamer --> Thread[Thread 시작: model.generate]

    Thread --> Loop{streamer.next 토큰?}
    Loop -- 있음 --> Yield[yield token.encode utf-8]
    Yield --> Loop
    Loop -- None --> Join[Thread.join 5s]
    Join --> Persist{텍스트 경로?}
    Persist -- yes --> AppendHist[history append user+assistant]
    Persist -- no --> Release
    AppendHist --> Release[clear_memory + lock.release]
    Release --> End([응답 종료])
```

---

## 9. 배포 다이어그램

```mermaid
flowchart LR
    subgraph DevMachine["개발자 머신 (Linux)"]
        subgraph Browser["Chromium 기반 브라우저"]
            UI["React App<br/>(Vite Dev Server :5173)"]
        end

        subgraph Backend["Python 프로세스"]
            UVI["uvicorn :8000"]
            API["FastAPI app<br/>(api.py)"]
            Core["core/ (singleton)<br/>ModelManager"]
            UVI --> API --> Core
        end

        subgraph GPU["NVIDIA RTX 4060 8GB"]
            Model["Mllama-3.2-Korean<br/>Bllossom-AICA-5B<br/>(NF4 4bit)"]
        end

        Core -. CUDA .-> Model
        UI -. "HTTP fetch<br/>+ ReadableStream" .-> UVI
    end

    subgraph HF["HuggingFace Hub (원격, 최초 1회)"]
        Repo["Bllossom/llama-3.2-Korean-Bllossom-AICA-5B"]
    end

    Core -. "최초 다운로드<br/>→ ~/.cache/huggingface" .-> Repo
```

---

## 10. 패키지 다이어그램

소스 트리의 패키지 의존 관계입니다.

```mermaid
flowchart TB
    subgraph frontend["frontend/src"]
        F_App["App.tsx"]
        F_Comp["components/<br/>ChatInput, ChatMessage"]
        F_Api["api/chat.ts"]
        F_App --> F_Comp
        F_App --> F_Api
    end

    subgraph backend["backend"]
        B_Api["api.py (FastAPI)"]
        B_Main["main.py (CLI 메뉴)"]
        subgraph backend_core["backend/core"]
            B_Cfg["config.py"]
            B_MM["model_manager.py"]
            B_TG["text_generator.py"]
            B_VG["vision_generator.py"]
        end
        subgraph backend_if["backend/interfaces"]
            B_CLI["cli_interface.py"]
        end

        B_Api --> B_Cfg
        B_Api --> B_MM
        B_Main --> B_Cfg
        B_Main --> B_MM
        B_Main --> B_TG
        B_Main --> B_VG
        B_CLI --> B_Cfg
        B_CLI --> B_MM
        B_CLI --> B_TG
        B_CLI --> B_VG
        B_MM --> B_Cfg
        B_TG --> B_MM
        B_TG --> B_Cfg
        B_VG --> B_MM
        B_VG --> B_Cfg
    end

    subgraph ext["외부 라이브러리"]
        E_HF["transformers<br/>(MllamaForConditionalGeneration<br/>MllamaProcessor<br/>TextIteratorStreamer)"]
        E_BNB["bitsandbytes<br/>(BitsAndBytesConfig)"]
        E_Torch["torch / CUDA"]
        E_PIL["Pillow + requests"]
        E_FastAPI["fastapi + uvicorn"]
    end

    F_Api -. HTTP .-> B_Api
    B_Api --> E_FastAPI
    B_MM --> E_HF
    B_MM --> E_BNB
    B_MM --> E_Torch
    B_VG --> E_PIL
```

---

## 부록 A. 다이어그램 갱신 가이드

다이어그램을 수정할 때:
1. **Mermaid 문법 검증** — VS Code의 *Markdown Preview Mermaid* 또는 https://mermaid.live 로 미리 검증.
2. **클래스 변경 시** — `1. 클래스 다이어그램`을 우선 갱신하고, 관련된 시퀀스 다이어그램의 participant 이름이 일치하는지 확인.
3. **API 추가 시** — `2. FastAPI Layer` + 신규 시퀀스 다이어그램 + `8. 활동 다이어그램`까지 점검.
4. **프론트엔드 컴포넌트 추가 시** — `3. 컴포넌트 다이어그램`에 박스를 추가하고 `App` 또는 부모 컴포넌트로부터의 `renders` 화살표를 그릴 것.
