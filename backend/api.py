"""FastAPI app for Korean Bllossom AICA-5B — single-user MVP.

Run from this directory:
    uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1

Streams plain UTF-8 text via HTTP chunked transfer; the frontend reads the
body as a ReadableStream and appends tokens incrementally.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Thread
from typing import AsyncIterator, List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from transformers import TextIteratorStreamer

from core.config import Config, setup_environment
from core.model_manager import ModelManager, get_model_manager

logger = logging.getLogger("bllossom.api")

# Skip the heavy model load when running the API just to validate routing
# or develop the frontend without GPU. Set MODEL_SKIP_LOAD=1 to enable.
SKIP_MODEL_LOAD = os.environ.get("MODEL_SKIP_LOAD") == "1"

# Per-user-process memory of the conversation. Text only; image turns are
# answered single-shot and not appended to history (multimodal history is
# fragile and out of scope for the MVP).
@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    text: str


@dataclass
class AppState:
    config: Optional[Config] = None
    manager: Optional[ModelManager] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    history: List[ChatTurn] = field(default_factory=list)


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_environment()
    state.config = Config(config_file="config.yaml")

    if SKIP_MODEL_LOAD:
        logger.warning("MODEL_SKIP_LOAD=1 — skipping model load; /api/chat will 503.")
    else:
        state.manager = get_model_manager(state.config)
        loaded = await asyncio.to_thread(state.manager.load_model)
        if not loaded:
            raise RuntimeError("Model failed to load — aborting startup.")
        logger.info("Model loaded; API ready.")

    try:
        yield
    finally:
        if state.manager and state.manager.is_loaded:
            try:
                state.manager.unload_model()
            except Exception:
                logger.exception("Error during model unload")


app = FastAPI(title="Bllossom Chat API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _build_text_messages(history: List[ChatTurn], prompt: str) -> list:
    messages = []
    for turn in history:
        messages.append(
            {"role": turn.role, "content": [{"type": "text", "text": turn.text}]}
        )
    messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    return messages


def _build_vision_messages(prompt: str) -> list:
    return [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]


def _resize_if_needed(image: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    if w > h:
        return image.resize((max_side, int(h * max_side / w)), Image.Resampling.LANCZOS)
    return image.resize((int(w * max_side / h), max_side), Image.Resampling.LANCZOS)


async def _stream_generation(
    prompt: str,
    image: Optional[Image.Image],
    max_new_tokens: int,
    temperature: float,
) -> AsyncIterator[str]:
    """Drive model.generate in a background thread and yield decoded tokens."""
    assert state.manager is not None
    manager = state.manager
    processor = manager.processor
    tokenizer = processor.tokenizer

    if image is None:
        messages = _build_text_messages(state.history, prompt)
    else:
        messages = _build_vision_messages(prompt)

    input_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        images=image,
        text=input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(manager.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=60.0,
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        use_cache=image is None,  # vision path disables KV cache (matches existing code)
        streamer=streamer,
    )
    if image is None:
        # Mllama's vision branch crashes RepetitionPenaltyLogitsProcessor
        # (scatter index OOB) on image+text inputs — only apply sampling
        # tweaks to the text path. Matches the working core/vision_generator.py.
        cfg = state.config.generation if state.config else None
        generation_kwargs.update(
            top_p=cfg.top_p if cfg else 0.9,
            top_k=cfg.top_k if cfg else 50,
            repetition_penalty=cfg.repetition_penalty if cfg else 1.1,
        )

    pieces: List[str] = []

    def _run_generate():
        try:
            with torch.no_grad():
                manager.model.generate(**generation_kwargs)
        except Exception:  # pragma: no cover — surfaces via streamer end
            logger.exception("generate() failed")

    thread = Thread(target=_run_generate, daemon=True)
    thread.start()

    loop = asyncio.get_running_loop()
    iterator = iter(streamer)

    try:
        while True:
            token = await loop.run_in_executor(None, lambda: next(iterator, None))
            if token is None:
                break
            pieces.append(token)
            yield token
    finally:
        # Let generate() finish before returning; daemon thread will also be
        # cleaned up on shutdown.
        await asyncio.to_thread(thread.join, 5.0)

    # Persist text-only turns in history
    if image is None:
        full = "".join(pieces).strip()
        if full:
            state.history.append(ChatTurn(role="user", text=prompt))
            state.history.append(ChatTurn(role="assistant", text=full))


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "model_loaded": bool(state.manager and state.manager.is_loaded),
        "history_turns": len(state.history),
    }


@app.post("/api/reset")
async def reset() -> dict:
    state.history.clear()
    return {"ok": True}


@app.post("/api/chat")
async def chat(
    prompt: str = Form(...),
    image: Optional[UploadFile] = File(None),
    max_new_tokens: int = Form(512),
    temperature: float = Form(0.7),
) -> StreamingResponse:
    if state.manager is None or not state.manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Empty prompt")

    pil_image: Optional[Image.Image] = None
    if image is not None:
        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty image")
        try:
            pil_image = Image.open(io.BytesIO(data))
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            pil_image = _resize_if_needed(pil_image)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Bad image: {exc}") from exc

    # Serialize requests — single GPU, single model instance
    if state.lock.locked():
        raise HTTPException(status_code=429, detail="A request is already in flight")

    async def stream() -> AsyncIterator[bytes]:
        async with state.lock:
            try:
                async for token in _stream_generation(
                    prompt=prompt,
                    image=pil_image,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                ):
                    yield token.encode("utf-8")
            except Exception as exc:
                logger.exception("stream failed")
                yield f"\n⚠️ 서버 오류: {exc}".encode("utf-8")
            finally:
                # clear_memory() touches CUDA; if a device-side assert poisoned
                # the context, this would raise a second exception that masks
                # the original. Swallow it — the lock release is what matters.
                if state.manager is not None:
                    try:
                        state.manager.clear_memory()
                    except Exception:
                        logger.exception("clear_memory() failed (ignored)")

    return StreamingResponse(
        stream(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
