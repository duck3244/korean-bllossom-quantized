export type Role = 'user' | 'assistant';

export interface Message {
  id: string;
  role: Role;
  content: string;
  imageUrl?: string | null;
}

interface Props {
  message: Message;
  streaming?: boolean;
}

export function ChatMessage({ message, streaming }: Props) {
  const isUser = message.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-2 text-sm leading-relaxed whitespace-pre-wrap ${
          isUser ? 'bg-neutral-900 text-white' : 'bg-white border border-neutral-200'
        }`}
      >
        {message.imageUrl && (
          <img
            src={message.imageUrl}
            alt=""
            className="mb-2 max-h-48 rounded object-contain"
          />
        )}
        {message.content}
        {streaming && <span className="ml-0.5 animate-pulse">▍</span>}
      </div>
    </div>
  );
}
