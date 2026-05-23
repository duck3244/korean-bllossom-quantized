const API_BASE = import.meta.env.VITE_API_BASE ?? '/api';

export async function resetChat(): Promise<void> {
  const res = await fetch(`${API_BASE}/reset`, { method: 'POST' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
}

export interface ChatRequest {
  prompt: string;
  image?: File | null;
}

export interface StreamHandlers {
  onToken: (delta: string) => void;
  onDone?: () => void;
  onError?: (err: unknown) => void;
  signal?: AbortSignal;
}

export async function streamChat(
  { prompt, image }: ChatRequest,
  { onToken, onDone, onError, signal }: StreamHandlers,
): Promise<void> {
  try {
    const form = new FormData();
    form.append('prompt', prompt);
    if (image) form.append('image', image);

    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      body: form,
      signal,
    });

    if (!res.ok || !res.body) {
      throw new Error(`HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      if (chunk) onToken(chunk);
    }
    onDone?.();
  } catch (err) {
    if ((err as { name?: string }).name === 'AbortError') return;
    onError?.(err);
  }
}
