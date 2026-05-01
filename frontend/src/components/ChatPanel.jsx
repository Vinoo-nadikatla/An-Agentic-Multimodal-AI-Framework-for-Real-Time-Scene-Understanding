// frontend/src/components/ChatPanel.jsx
// Displays conversation history, streaming response, and a text input box.

import { useEffect, useRef, useState } from "react";

export default function ChatPanel({ messages, streamingText, isThinking, onSend, connected }) {
  const [draft, setDraft] = useState("");
  const bottomRef = useRef(null);

  // Auto-scroll on new content
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  const handleSend = () => {
    const text = draft.trim();
    if (!text) return;
    setDraft("");
    onSend(text);
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-panel">
      {/* Status bar */}
      <div className="chat-status">
        <span className={`status-dot ${connected ? "online" : "offline"}`} />
        <span className="status-label">{connected ? "Connected" : "Reconnecting…"}</span>
      </div>

      {/* Message list */}
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-bubble ${msg.role}`}>
            <p>{msg.text}</p>
          </div>
        ))}

        {/* Streaming assistant message */}
        {(isThinking || streamingText) && (
          <div className="chat-bubble assistant streaming">
            {streamingText
              ? <p>{streamingText}</p>
              : <span className="thinking-dots"><span /><span /><span /></span>
            }
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="chat-input-bar">
        <textarea
          className="chat-input"
          rows={1}
          placeholder={connected ? "Ask about what you see…" : "Connecting…"}
          value={draft}
          onChange={e => setDraft(e.target.value)}
          onKeyDown={handleKey}
          disabled={!connected || isThinking}
        />
        <button
          className="send-btn"
          onClick={handleSend}
          disabled={!connected || isThinking || !draft.trim()}
        >
          ↑
        </button>
      </div>
    </div>
  );
}
