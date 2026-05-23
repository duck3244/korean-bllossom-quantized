import { useRef, useState, type KeyboardEvent } from 'react';

interface Props {
  disabled?: boolean;
  onSubmit: (text: string, image: File | null) => void;
  onCancel?: () => void;
  busy?: boolean;
}

export function ChatInput({ disabled, onSubmit, onCancel, busy }: Props) {
  const [text, setText] = useState('');
  const [image, setImage] = useState<File | null>(null);
  const previewRef = useRef<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const previewUrl = (() => {
    if (!image) {
      if (previewRef.current) {
        URL.revokeObjectURL(previewRef.current);
        previewRef.current = null;
      }
      return null;
    }
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    previewRef.current = URL.createObjectURL(image);
    return previewRef.current;
  })();

  const submit = () => {
    const trimmed = text.trim();
    if (!trimmed) return;
    onSubmit(trimmed, image);
    setText('');
    setImage(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // 한글 IME 조합 중 Enter는 전송하지 않음 — 한국어 챗 #1 footgun 방지
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="border-t border-neutral-200 bg-white p-3">
      {previewUrl && (
        <div className="mb-2 flex items-center gap-2">
          <img
            src={previewUrl}
            alt="첨부 이미지"
            className="h-16 w-16 rounded object-cover border border-neutral-200"
          />
          <button
            type="button"
            className="text-sm text-neutral-500 hover:text-neutral-800"
            onClick={() => setImage(null)}
          >
            제거
          </button>
        </div>
      )}
      <div className="flex items-end gap-2">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => setImage(e.target.files?.[0] ?? null)}
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          className="h-10 rounded-md border border-neutral-300 px-3 text-sm hover:bg-neutral-100 disabled:opacity-50"
          title="이미지 첨부"
        >
          이미지
        </button>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={disabled}
          rows={1}
          placeholder="메시지를 입력하세요 (Shift+Enter 줄바꿈)"
          className="min-h-[40px] max-h-40 flex-1 resize-none rounded-md border border-neutral-300 px-3 py-2 text-sm focus:border-neutral-500 focus:outline-none disabled:opacity-50"
        />
        {busy ? (
          <button
            type="button"
            onClick={onCancel}
            className="h-10 rounded-md bg-red-600 px-4 text-sm font-medium text-white hover:bg-red-700"
          >
            중단
          </button>
        ) : (
          <button
            type="button"
            onClick={submit}
            disabled={disabled || !text.trim()}
            className="h-10 rounded-md bg-neutral-900 px-4 text-sm font-medium text-white hover:bg-neutral-800 disabled:opacity-40"
          >
            전송
          </button>
        )}
      </div>
    </div>
  );
}
