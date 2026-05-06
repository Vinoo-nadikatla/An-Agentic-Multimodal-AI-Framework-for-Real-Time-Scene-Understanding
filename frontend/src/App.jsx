import { useEffect, useRef, useState } from "react";
import { useWebSocket } from "./hooks/useWebSocket";
import "./App.css";

const SESSION_ID = crypto.randomUUID();
const now = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

const SUGGESTIONS = [
  "What do you see right now?",
  "How many people are in the room?",
  "Describe my surroundings",
  "Is there any movement?",
];

function speak(text) {
  if (!text || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();

  const clean = text.replace(/[*_#`]/g, "").trim();
  if (!clean) return;

  const utt = new SpeechSynthesisUtterance(clean);
  utt.rate = 0.92;
  utt.pitch = 1;
  utt.volume = 1;

  const hasTelugu = /[\u0C00-\u0C7F]/.test(clean);
  const hasHindi  = /[\u0900-\u097F]/.test(clean);
  const hasTamil  = /[\u0B80-\u0BFF]/.test(clean);

  if (hasTelugu)      utt.lang = "te-IN";
  else if (hasHindi)  utt.lang = "hi-IN";
  else if (hasTamil)  utt.lang = "ta-IN";
  else                utt.lang = "en-US";

  const trySpeak = (attempts) => {
    const voices = window.speechSynthesis.getVoices();
    if (voices.length === 0 && attempts > 0) {
      setTimeout(() => trySpeak(attempts - 1), 250);
      return;
    }
    const exact   = voices.find(v => v.lang === utt.lang);
    const partial = voices.find(v => v.lang.startsWith(utt.lang.split("-")[0]));
    if (exact)        utt.voice = exact;
    else if (partial) utt.voice = partial;
    window.speechSynthesis.speak(utt);
  };

  trySpeak(6);
}

export default function App() {
  const [messages, setMessages]     = useState([]);
  const [streaming, setStreaming]   = useState("");
  const [isThinking, setThinking]   = useState(false);
  const [draft, setDraft]           = useState("");
  const [sceneInfo, setScene]       = useState(null);
  const [recording, setRecording]   = useState(false);
  const [processing, setProcessing] = useState(false);
  const [ttsOn, setTtsOn]           = useState(true);
  const [camStream, setCamStream]   = useState(null);

  const bottomRef   = useRef(null);
  const inputRef    = useRef(null);
  const recorderRef = useRef(null);
  const chunksRef   = useRef([]);
  const videoRef    = useRef(null);  

  const API_BASE = import.meta.env.VITE_API_URL || "";
  const WS_BASE  = API_BASE.replace(/^http/, "ws") || `ws://${window.location.host}`;

  const { sendMessage, connected } = useWebSocket(`${WS_BASE}/ws/${SESSION_ID}`, {
    onToken:  (t) => setStreaming((p) => p + t),
    onStart:  ()  => { setThinking(true); setStreaming(""); },
    onEnd:    ()  => {
      setThinking(false);
      setStreaming((prev) => {
        if (prev.trim()) {
          setMessages((m) => {
            const msg = { role: "assistant", text: prev, time: now() };
            if (ttsOn) speak(prev);
            return [...m, msg];
          });
        }
        return "";
      });
    },
    onError: (text) => {
      setThinking(false);
      setStreaming("");
      setMessages((m) => [...m, { role: "assistant", text, time: now() }]);
    },
  });

  // scene polling
  useEffect(() => {
    const poll = async () => {
      try { const r = await fetch(`${API_BASE}/api/scene`); if (r.ok) setScene(await r.json()); }
      catch (_) {}
    };
    poll();
    const id = setInterval(poll, 3000);
    return () => clearInterval(id);
  }, []);

  // Send browser camera frames to backend for AI vision
  useEffect(() => {
    let alive = true;
    let stream = null;
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    const startCapture = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
          audio: false
        });
        const video = document.createElement("video");
        video.srcObject = stream;
        await video.play();

        const sendFrame = async () => {
          if (!alive) return;
          canvas.width = 512;
          canvas.height = 512;
          ctx.drawImage(video, 0, 0, 512, 512);
          const b64 = canvas.toDataURL("image/jpeg", 0.7).split(",")[1];
          try {
            await fetch(`${API_BASE}/api/stream-frame`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ session_id: SESSION_ID, frame_b64: b64 })
            });
          } catch (_) {}
          if (alive) setTimeout(sendFrame, 2000);
        };
        sendFrame();
      } catch (e) {
        console.log("Browser camera not available:", e.message);
      }
    };

    startCapture();
    return () => {
      alive = false;
      stream?.getTracks().forEach(t => t.stop());
    };
  }, []);

  // auto scroll
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, streaming]);

  // camera preview — polls server-side OpenCV feed
  useEffect(() => {
    let alive = true;
    const loop = async () => {
      while (alive) {
        try {
          const r = await fetch(`${API_BASE}/api/camera-feed`);
          if (r.ok) {
            const d = await r.json();
            if (d.frame) setCamStream(d.frame);
          }
        } catch (_) {}
        await new Promise(res => setTimeout(res, 500));
      }
    };
    loop();
    return () => { alive = false; };
  }, []);

  const send = (text) => {
    const t = text.trim();
    if (!t || !connected || isThinking) return;
    setMessages((m) => [...m, { role: "user", text: t, time: now() }]);
    sendMessage({ type: "message", text: t });
    setDraft("");
    inputRef.current?.focus();
  };

  const startRec = async () => {
    chunksRef.current = [];
    try {
      const stream   = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      recorderRef.current = recorder;
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setProcessing(true);
        try {
          const blob = new Blob(chunksRef.current, { type: "audio/webm" });
          const form = new FormData();
          form.append("file", blob, "voice.webm");
          const resp = await fetch(`${API_BASE}/api/transcribe`, { method: "POST", body: form });
          if (resp.ok) { const { text } = await resp.json(); if (text?.trim()) send(text); }
        } finally { setProcessing(false); }
      };
      recorder.start(); setTimeout(() => {}, 500);
      setRecording(true);
    } catch (e) { console.error(e); }
  };

  const stopRec = () => { if (!recording) return; recorderRef.current?.stop(); setRecording(false); };

  return (
    <div className="shell">
      <div className="bg-mesh" />

      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="brand">
          <svg className="brand-icon" viewBox="0 0 32 32" fill="none">
            <circle cx="16" cy="16" r="14" stroke="url(#g1)" strokeWidth="2"/>
            <circle cx="16" cy="16" r="6" fill="url(#g2)"/>
            <circle cx="16" cy="7" r="2" fill="#a78bfa"/>
            <circle cx="25" cy="21" r="2" fill="#60a5fa"/>
            <circle cx="7" cy="21" r="2" fill="#f472b6"/>
            <defs>
              <linearGradient id="g1" x1="0" y1="0" x2="32" y2="32">
                <stop offset="0%" stopColor="#a78bfa"/><stop offset="100%" stopColor="#60a5fa"/>
              </linearGradient>
              <linearGradient id="g2" x1="0" y1="0" x2="32" y2="32">
                <stop offset="0%" stopColor="#818cf8"/><stop offset="100%" stopColor="#38bdf8"/>
              </linearGradient>
            </defs>
          </svg>
          <span className="brand-name">VN AI</span>
        </div>

        {/* Camera preview */}
        <div className="cam-preview-wrap">
          <div className="cam-preview">
            {camStream
              ? <img
                  src={`data:image/jpeg;base64,${camStream}`}
                  className="cam-video"
                  alt="Live camera"
                  style={{width:"100%",height:"100%",objectFit:"cover",display:"block"}}
                />
              : <div className="cam-no-feed">
                  <span>📷</span>
                  <p>Camera loading...</p>
                </div>
            }
            <div className={`cam-live-badge ${camStream ? "live" : ""}`}>
              <span className="live-dot"/> {camStream ? "LIVE" : "OFF"}
            </div>
          </div>
        </div>

        {sceneInfo && (
          <div className="sidebar-section">
            <p className="sidebar-label">Scene Analysis</p>
            <div className="scene-card">
              <div className="scene-row">
                <span className="scene-icon">👥</span>
                <span className="scene-val">{sceneInfo.people} {sceneInfo.people === 1 ? "Person" : "People"} detected</span>
              </div>
              <div className="scene-row">
                <span className="scene-icon">{sceneInfo.motion ? "⚡" : "○"}</span>
                <span className="scene-val">{sceneInfo.motion ? "Motion Detected" : "No Motion"}</span>
              </div>
            </div>
          </div>
        )}

        <div className="sidebar-section">
          <p className="sidebar-label">Quick Ask</p>
          <div className="quick-btns">
            {SUGGESTIONS.map((s) => (
              <button key={s} className="quick-btn" onClick={() => send(s)}>{s}</button>
            ))}
          </div>
        </div>

        <div className="tts-toggle">
          <span>🔊 Voice replies</span>
          <button className={`toggle-btn ${ttsOn ? "on" : ""}`} onClick={() => setTtsOn(!ttsOn)}>
            {ttsOn ? "ON" : "OFF"}
          </button>
        </div>

        <div className="sidebar-footer">Powered by Groq + LangGraph</div>
      </aside>

      {/* ── Chat ── */}
      <div className="chat-panel">
        <div className="topbar">
          <span className="topbar-title">Vision Assistant</span>
          <span className={`conn-badge ${connected ? "on" : "off"}`}>
            <span className="conn-dot"/> {connected ? "Connected" : "Reconnecting…"}
          </span>
        </div>

        <div className="messages-area">
          {messages.length === 0 && !isThinking && (
            <div className="welcome">
              <div className="welcome-glow"/>
              <h1>Hello! 👋</h1>
              <p>I'm your AI vision assistant. I can see through your camera and answer questions about your environment in real time.</p>
              <div className="welcome-chips">
                {SUGGESTIONS.map((s) => (
                  <button key={s} className="welcome-chip" onClick={() => send(s)}>{s}</button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`msg ${msg.role}`}>
              <div className="msg-avatar">
                {msg.role === "assistant"
                  ? <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" fill="url(#av)"/><circle cx="12" cy="12" r="4" fill="white" fillOpacity=".9"/><defs><linearGradient id="av" x1="0" y1="0" x2="24" y2="24"><stop stopColor="#818cf8"/><stop offset="1" stopColor="#38bdf8"/></linearGradient></defs></svg>
                  : <span>U</span>
                }
              </div>
              <div className="msg-content">
                <div className={`msg-bubble ${msg.role}`}>{msg.text}</div>
                <div className="msg-meta">
                  <span className="msg-time">{msg.time}</span>
                  {msg.role === "assistant" && (
                    <button className="replay-btn" onClick={() => speak(msg.text)} title="Replay audio">🔊</button>
                  )}
                </div>
              </div>
              {msg.role === "user" && <div className="avatar user-avatar">U</div>}
            </div>
          ))}

          {(isThinking || streaming) && (
            <div className="msg assistant">
              <div className="msg-avatar">
                <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" fill="url(#av2)"/><circle cx="12" cy="12" r="4" fill="white" fillOpacity=".9"/><defs><linearGradient id="av2" x1="0" y1="0" x2="24" y2="24"><stop stopColor="#818cf8"/><stop offset="1" stopColor="#38bdf8"/></linearGradient></defs></svg>
              </div>
              <div className="msg-content">
                <div className="msg-bubble assistant streaming">
                  {streaming
                    ? <>{streaming}<span className="blink-cursor"/></>
                    : <div className="dots"><span/><span/><span/></div>
                  }
                </div>
              </div>
            </div>
          )}
          <div ref={bottomRef}/>
        </div>

        <div className="input-bar">
          <button
            className={`voice-btn ${recording ? "rec" : ""} ${processing ? "proc" : ""}`}
            onClick={() => recording ? stopRec() : startRec()}
            onTouchStart={(e) => { e.preventDefault(); startRec(); }}
            onTouchEnd={(e)   => { e.preventDefault(); stopRec(); }}
            disabled={processing || isThinking}
            title="Hold to speak"
          >
            {processing ? <span className="spin">◌</span> : recording ? "■" : "🎙"}
          </button>

          <textarea
            ref={inputRef}
            className="chat-input"
            rows={1}
            placeholder={connected ? "Type a message…  (Enter to send)" : "Connecting…"}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(draft); }}}
            disabled={!connected || isThinking}
          />

          <button
            className={`send-btn ${draft.trim() && connected ? "active" : ""}`}
            onClick={() => send(draft)}
            disabled={!connected || isThinking || !draft.trim()}
          >↑</button>
        </div>
      </div>
    </div>
  );
}
