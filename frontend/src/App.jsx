import { useEffect, useRef, useState } from "react";
import { useWebSocket } from "./hooks/useWebSocket";
import "./App.css";

const SESSION_ID = crypto.randomUUID();
const now = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

const SUGGESTIONS = [
  "Check PPE compliance now",
  "How many workers are present?",
  "Describe safety hazards you see",
  "Is anyone missing a helmet?",
];

async function speakWithGTTS(text, apiBase, setCurrentAudio) {
  try {
    const resp = await fetch(`${apiBase}/api/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    if (!resp.ok) throw new Error("TTS failed");
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    setCurrentAudio(audio);
    audio.onended = () => { URL.revokeObjectURL(url); setCurrentAudio(null); };
    await audio.play();
  } catch (e) {
    console.error("gTTS failed:", e);
  }
}

function speak(text, apiBase = "", setCurrentAudio = () => {}) {
  if (!text) return;
  const clean = text.replace(/[*_#`]/g, "").trim();
  if (!clean) return;

  const hasTelugu = /[ఀ-౿]/.test(clean);
  const hasHindi  = /[ऀ-ॿ]/.test(clean);
  const hasTamil  = /[஀-௿]/.test(clean);

  if (hasTelugu || hasHindi || hasTamil) {
    speakWithGTTS(clean, apiBase, setCurrentAudio);
    return;
  }

  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(clean);
  utt.rate = 0.92;
  utt.lang = "en-US";
  window.speechSynthesis.speak(utt);
}

function complianceColor(pct) {
  if (pct >= 80) return "#00C851";
  if (pct >= 50) return "#FF8800";
  return "#FF4444";
}

function workerIcon(status) {
  if (status === "compliant") return "✅";
  if (status === "partial")   return "⚠️";
  return "❌";
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
  const [currentAudio, setCurrentAudio] = useState(null);
  const [ppeStatus, setPpeStatus]   = useState(null);
  const [safetyAlerts, setSafetyAlerts] = useState({ alerts: [], total: 0 });

  const bottomRef        = useRef(null);
  const inputRef         = useRef(null);
  const recorderRef      = useRef(null);
  const chunksRef        = useRef([]);
  const recordingStartRef = useRef(null);

  const API_BASE = import.meta.env.VITE_API_URL || "";
  const WS_BASE  = API_BASE.replace(/^http/, "ws") || `ws://${window.location.host}`;

  const { sendMessage, connected } = useWebSocket(`${WS_BASE}/ws/${SESSION_ID}`, {
    onToken:  (t) => setStreaming((p) => p + t),
    onStart:  ()  => { setThinking(true); setStreaming(""); },
    onEnd:    ()  => {
      setThinking(false);
      setStreaming((prev) => {
        if (prev) {
          setMessages((m) => [...m, { role: "assistant", text: prev, time: now() }]);
          if (ttsOn) speak(prev, API_BASE, setCurrentAudio);
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

  // Scene polling
  useEffect(() => {
    const poll = async () => {
      try { const r = await fetch(`${API_BASE}/api/scene`); if (r.ok) setScene(await r.json()); }
      catch (_) {}
    };
    poll();
    const id = setInterval(poll, 5000);
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
          canvas.width = 320;
          canvas.height = 320;
          ctx.drawImage(video, 0, 0, 320, 320);
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

  // Slow PPE status poll — compliance numbers, worker list
  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/ppe-status`);
        const data = await res.json();
        setPpeStatus(data);
      } catch (_) {}
    };
    poll();
    const id = setInterval(poll, 3000);
    return () => clearInterval(id);
  }, []);

  // Safety alerts — poll every 8 seconds
  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch(`${API_BASE}/api/safety-alerts`);
        if (r.ok) setSafetyAlerts(await r.json());
      } catch (_) {}
    };
    poll();
    const id = setInterval(poll, 8000);
    return () => clearInterval(id);
  }, []);

  // Automated TTS safety announcements — speak when violation exceeds 30s
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/announcements`);
        const data = await res.json();
        if (data.announcements?.length > 0) {
          data.announcements.forEach(text => {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.9;
            utterance.volume = 1;
            window.speechSynthesis.speak(utterance);
          });
        }
      } catch (_) {}
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  // Auto scroll
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, streaming]);

  const send = (text) => {
    const t = text.trim();
    if (!t || !connected || isThinking) return;
    window.speechSynthesis?.cancel();
    setCurrentAudio(prev => { prev?.pause(); return null; });
    setMessages((m) => [...m, { role: "user", text: t, time: now() }]);
    sendMessage({ type: "message", text: t });
    setDraft("");
    inputRef.current?.focus();
  };

  const startRec = async () => {
    window.speechSynthesis?.cancel();
    setCurrentAudio(prev => { prev?.pause(); return null; });
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
          if (blob.size < 1000) {
            console.warn("Audio too short or empty, skipping transcription");
            setMessages((m) => [...m, { role: "assistant", text: "No audio detected, please try again.", time: now() }]);
            return;
          }
          const form = new FormData();
          form.append("file", blob, "voice.webm");
          const resp = await fetch(`${API_BASE}/api/transcribe`, { method: "POST", body: form });
          if (resp.ok) { const { text } = await resp.json(); if (text?.trim()) send(text); }
        } finally { setProcessing(false); }
      };
      recorder.start();
      recordingStartRef.current = Date.now();
      setRecording(true);
    } catch (e) { console.error(e); }
  };

  const stopRec = async () => {
    if (!recording) return;
    const MIN_RECORD_MS = 500;
    const elapsed = Date.now() - (recordingStartRef.current ?? Date.now());
    if (elapsed < MIN_RECORD_MS) {
      await new Promise((resolve) => setTimeout(resolve, MIN_RECORD_MS - elapsed));
    }
    recorderRef.current?.stop();
    setRecording(false);
  };

  const overallPct   = ppeStatus?.overall_compliance ?? 0;
  const helmetPct    = ppeStatus?.helmet_compliance  ?? 0;
  const totalWorkers = ppeStatus?.total_workers ?? 0;
  const compliantCnt = ppeStatus?.compliant ?? 0;
  const workers      = ppeStatus?.workers ?? [];
  const modelReady   = ppeStatus?.model_ready ?? false;
  const workersPct   = totalWorkers > 0 ? Math.round(compliantCnt / totalWorkers * 100) : 0;
  const workersColor = compliantCnt === totalWorkers && totalWorkers > 0 ? "#00C851" : "#FF4444";

  return (
    <div className="shell">
      <div className="bg-mesh" />

      {/* ── LEFT PANEL: Chat ── */}
      <div className="chat-panel">
        <div className="topbar">
          <div className="topbar-left">
            <svg className="brand-icon" viewBox="0 0 32 32" fill="none">
              <circle cx="16" cy="16" r="14" stroke="url(#g1)" strokeWidth="2"/>
              <circle cx="16" cy="16" r="6" fill="url(#g2)"/>
              <circle cx="16" cy="7"  r="2" fill="#a78bfa"/>
              <circle cx="25" cy="21" r="2" fill="#60a5fa"/>
              <circle cx="7"  cy="21" r="2" fill="#f472b6"/>
              <defs>
                <linearGradient id="g1" x1="0" y1="0" x2="32" y2="32">
                  <stop offset="0%" stopColor="#a78bfa"/><stop offset="100%" stopColor="#60a5fa"/>
                </linearGradient>
                <linearGradient id="g2" x1="0" y1="0" x2="32" y2="32">
                  <stop offset="0%" stopColor="#818cf8"/><stop offset="100%" stopColor="#38bdf8"/>
                </linearGradient>
              </defs>
            </svg>
            <span className="topbar-title">VN AI Safety Monitor</span>
          </div>
          <div className="topbar-right">
            <button className={`toggle-btn ${ttsOn ? "on" : ""}`} onClick={() => setTtsOn(!ttsOn)}
              title="Toggle voice replies">
              {ttsOn ? "🔊 ON" : "🔇 OFF"}
            </button>
            <span className={`conn-badge ${connected ? "on" : "off"}`}>
              <span className="conn-dot"/> {connected ? "Live" : "Reconnecting…"}
            </span>
          </div>
        </div>

        <div className="messages-area">
          {messages.length === 0 && !isThinking && (
            <div className="welcome">
              <div className="welcome-glow"/>
              <h1>Safety Monitor</h1>
              <p>I watch your workspace in real time. Ask me about PPE compliance, worker safety, or what I see on camera.</p>
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
                    <button className="replay-btn" onClick={() => speak(msg.text, API_BASE, setCurrentAudio)} title="Replay audio">🔊</button>
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
            placeholder={connected ? "Ask about safety…  (Enter to send)" : "Connecting…"}
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

      {/* ── RIGHT PANEL: Safety Monitoring ── */}
      <div className="safety-panel">

        {/* Alert banner */}
        {modelReady && (
          <div className={`alert-banner ${safetyAlerts.total > 0 ? "danger" : "safe"}`}>
            {safetyAlerts.total > 0 ? (
              <>
                <span className="alert-icon">🚨</span>
                <span className="alert-text">
                  <strong>{safetyAlerts.total} violation{safetyAlerts.total > 1 ? "s" : ""} detected — </strong>
                  {safetyAlerts.alerts.map(a =>
                    `${a.worker}: ${a.violations.join(", ")}`
                  ).join(" · ")}
                </span>
              </>
            ) : (
              <>
                <span className="alert-icon">✅</span>
                <span className="alert-text"><strong>All workers compliant</strong> — no PPE violations</span>
              </>
            )}
          </div>
        )}

        {/* TOP: Live annotated camera feed */}
        <div className="ppe-feed-section">
          <div className="section-header">
            <span>Live Camera Feed</span>
            <div className="cam-live-badge live">
              <span className="live-dot"/> LIVE
            </div>
          </div>
          <div className="ppe-feed-wrap">
            <img
              src={`${API_BASE}/api/video-stream`}
              className="ppe-feed-img"
              alt="Live PPE feed"
              style={{ width: "100%", height: "auto", display: "block" }}
            />
            {!ppeStatus?.model_ready && (
              <div style={{
                position: "absolute", bottom: "8px", right: "8px",
                background: "rgba(0,0,0,0.7)", color: "white",
                padding: "4px 8px", borderRadius: "4px", fontSize: "12px",
                pointerEvents: "none",
              }}>
                PPE model loading…
              </div>
            )}
          </div>
        </div>

        {/* BOTTOM: Compliance Dashboard */}
        <div className="compliance-section">
          <div className="section-header">
            <span>PPE Compliance Dashboard</span>
            {sceneInfo && (
              <span className="scene-pill">
                👥 {sceneInfo.people} {sceneInfo.people === 1 ? "person" : "people"}
                {sceneInfo.motion ? " · ⚡ motion" : ""}
              </span>
            )}
          </div>

          <div className="compliance-body">
            {/* Large compliance score */}
            <div className="score-block">
              <div className="score-ring" style={{ "--score-color": complianceColor(overallPct) }}>
                <span className="score-number" style={{ color: complianceColor(overallPct) }}>
                  {overallPct}%
                </span>
                <span className="score-label">Overall</span>
              </div>
            </div>

            {/* Metric rows */}
            <div className="metrics-block">
              <div className="metric-row">
                <span className="metric-icon">⛑️</span>
                <span className="metric-name">Helmet</span>
                <div className="metric-bar-wrap">
                  <div className="metric-bar" style={{
                    width: `${helmetPct}%`,
                    background: complianceColor(helmetPct)
                  }}/>
                </div>
                <span className="metric-pct" style={{ color: complianceColor(helmetPct) }}>
                  {helmetPct}%
                </span>
              </div>

              <div className="metric-row">
                <span className="metric-icon">👷</span>
                <span className="metric-name">Workers</span>
                <div className="metric-bar-wrap">
                  <div className="metric-bar" style={{
                    width: `${workersPct}%`,
                    background: workersColor,
                  }}/>
                </div>
                <span className="metric-pct" style={{ color: workersColor }}>
                  {compliantCnt}/{totalWorkers}
                </span>
              </div>
            </div>
          </div>

          {/* Per-worker list */}
          <div className="worker-list">
            {workers.length === 0 ? (
              <div className="worker-empty">
                {modelReady
                  ? "No workers detected in frame"
                  : "PPE model initializing — visual detection starting soon"}
              </div>
            ) : (
              workers.map((w) => (
                <div key={w.id} className={`worker-item ${w.status}`}>
                  <span className="worker-icon">{workerIcon(w.status)}</span>
                  <span className="worker-label">{w.label}</span>
                  <span className="worker-status">
                    {w.violations.length === 0
                      ? "Compliant"
                      : w.violations.join(" · ")}
                  </span>
                  <div className="worker-badges">
                    <span className={`ppe-badge ${w.has_helmet ? "ok" : "missing"}`}>
                      ⛑️ {w.has_helmet ? "Helmet" : "No Helmet"}
                    </span>
                    {w.violation_duration && (
                      <span style={{ color: "#FF4444", fontSize: "11px", marginLeft: "8px" }}>
                        ⏱ {w.violation_duration}
                      </span>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
