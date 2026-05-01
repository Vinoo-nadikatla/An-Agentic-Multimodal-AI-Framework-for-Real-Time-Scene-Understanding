// frontend/src/components/AudioRecorder.jsx
// Hold-to-talk voice input. Records via MediaRecorder, POSTs to /api/transcribe.

import { useRef, useState } from "react";

export default function AudioRecorder({ onTranscript, apiBase, disabled }) {
  const [recording, setRecording]   = useState(false);
  const [processing, setProcessing] = useState(false);
  const recorderRef  = useRef(null);
  const chunksRef    = useRef([]);

  const startRecording = async () => {
    if (disabled || recording) return;
    chunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      recorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        setProcessing(true);
        try {
          const blob = new Blob(chunksRef.current, { type: "audio/webm" });
          const form = new FormData();
          form.append("file", blob, "voice.webm");

          const resp = await fetch(`${apiBase}/api/transcribe`, {
            method: "POST",
            body: form,
          });

          if (resp.ok) {
            const { text } = await resp.json();
            if (text?.trim()) onTranscript?.(text);
          }
        } catch (e) {
          console.error("Transcription error:", e);
        } finally {
          setProcessing(false);
        }
      };

      recorder.start();
      setRecording(true);
    } catch (e) {
      console.error("Mic error:", e);
    }
  };

  const stopRecording = () => {
    if (!recording) return;
    recorderRef.current?.stop();
    setRecording(false);
  };

  const label = processing ? "Transcribing…"
              : recording  ? "Release to send"
              : disabled   ? "Agent thinking…"
              : "Hold to speak";

  return (
    <div className="audio-recorder">
      <button
        className={`mic-btn ${recording ? "recording" : ""} ${processing || disabled ? "muted" : ""}`}
        onMouseDown={startRecording}
        onMouseUp={stopRecording}
        onTouchStart={e => { e.preventDefault(); startRecording(); }}
        onTouchEnd={e => { e.preventDefault(); stopRecording(); }}
        disabled={processing || disabled}
      >
        {recording
          ? <span className="mic-icon pulse">⏹</span>
          : <span className="mic-icon">🎙</span>
        }
      </button>
      <span className="mic-label">{label}</span>
    </div>
  );
}
