import { useEffect, useRef, useState } from 'react';
import { ChatInput } from './components/ChatInput';
import { ChatMessage, type Message } from './components/ChatMessage';
import { streamChat, resetChat } from './api/chat';

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  const send = async (text: string, image: File | null) => {
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      imageUrl: image ? URL.createObjectURL(image) : null,
    };
    const assistantMsg: Message = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '',
    };
    setMessages((m) => [...m, userMsg, assistantMsg]);
    setBusy(true);

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    await streamChat(
      { prompt: text, image },
      {
        signal: ctrl.signal,
        onToken: (delta) =>
          setMessages((m) =>
            m.map((msg) =>
              msg.id === assistantMsg.id
                ? { ...msg, content: msg.content + delta }
                : msg,
            ),
          ),
        onError: (err) =>
          setMessages((m) =>
            m.map((msg) =>
              msg.id === assistantMsg.id
                ? { ...msg, content: `⚠️ 오류: ${String(err)}` }
                : msg,
            ),
          ),
      },
    );
    setBusy(false);
    abortRef.current = null;
  };

  const cancel = () => abortRef.current?.abort();

  const newChat = async () => {
    if (busy) cancel();
    try {
      await resetChat();
    } catch {
      // 백엔드가 꺼져 있어도 클라이언트 상태는 비움
    }
    setMessages([]);
  };

  return (
    <div className="flex h-full flex-col bg-neutral-50">
      <header className="flex items-center justify-between border-b border-neutral-200 bg-white px-4 py-3">
        <div>
          <h1 className="text-base font-semibold">Bllossom Chat</h1>
          <p className="text-xs text-neutral-500">Korean Bllossom AICA-5B · MVP</p>
        </div>
        <button
          type="button"
          onClick={newChat}
          disabled={messages.length === 0 && !busy}
          className="rounded-md border border-neutral-300 px-3 py-1.5 text-sm hover:bg-neutral-100 disabled:opacity-40"
        >
          새 대화
        </button>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {messages.length === 0 && (
          <div className="mt-20 text-center text-sm text-neutral-400">
            대화를 시작해보세요.
          </div>
        )}
        {messages.map((msg, i) => (
          <ChatMessage
            key={msg.id}
            message={msg}
            streaming={busy && i === messages.length - 1 && msg.role === 'assistant'}
          />
        ))}
      </div>

      <ChatInput onSubmit={send} onCancel={cancel} busy={busy} />
    </div>
  );
}
