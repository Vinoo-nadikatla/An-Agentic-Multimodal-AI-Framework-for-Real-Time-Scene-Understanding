// frontend/src/hooks/useWebSocket.js
// Manages a WebSocket connection with auto-reconnect and keepalive.

import { useCallback, useEffect, useRef, useState } from "react";

const RECONNECT_DELAY_MS = 2000;
const MAX_RECONNECTS = 10;

export function useWebSocket(url, { onToken, onStart, onEnd, onError } = {}) {
  const [connected, setConnected] = useState(false);
  const wsRef        = useRef(null);
  const reconnectRef = useRef(0);
  const mountedRef   = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      reconnectRef.current = 0;
    };

    ws.onmessage = (evt) => {
      let msg;
      try { msg = JSON.parse(evt.data); } catch { return; }

      switch (msg.type) {
        case "token":  onToken?.(msg.text); break;
        case "start":  onStart?.(); break;
        case "end":    onEnd?.(); break;
        case "error":  onError?.(msg.text); break;
        case "ping":
          // respond to server keepalive
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "pong" }));
          }
          break;
        default: break;
      }
    };

    ws.onclose = () => {
      setConnected(false);
      if (!mountedRef.current) return;
      if (reconnectRef.current < MAX_RECONNECTS) {
        reconnectRef.current += 1;
        setTimeout(connect, RECONNECT_DELAY_MS);
      }
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [url]);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      wsRef.current?.close();
    };
  }, [connect]);

  const sendMessage = useCallback((payload) => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }, []);

  return { connected, sendMessage };
}
