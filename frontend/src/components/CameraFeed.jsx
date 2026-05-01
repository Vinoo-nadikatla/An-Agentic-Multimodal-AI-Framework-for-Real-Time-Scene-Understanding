// frontend/src/components/CameraFeed.jsx
// Captures webcam via WebRTC, shows preview, and streams frames to the backend.

import { useEffect, useRef, useState } from "react";

const FRAME_INTERVAL_MS = 500; // stream a frame every 500ms (2 fps)
const FRAME_SIZE = 512;        // resize before sending to save bandwidth

export default function CameraFeed({ sessionId, apiBase, sceneInfo }) {
  const videoRef   = useRef(null);
  const canvasRef  = useRef(null);
  const intervalRef = useRef(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [error, setError]       = useState(null);

  // ── Start camera ────────────────────────────────────────────────────────
  const startCamera = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setCameraOn(true);
      }
    } catch (e) {
      setError("Camera access denied. Please allow camera permissions.");
    }
  };

  const stopCamera = () => {
    const stream = videoRef.current?.srcObject;
    stream?.getTracks().forEach(t => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
    clearInterval(intervalRef.current);
  };

  // ── Frame streaming loop ────────────────────────────────────────────────
  useEffect(() => {
    if (!cameraOn) return;

    const canvas = canvasRef.current;
    const ctx    = canvas.getContext("2d");

    intervalRef.current = setInterval(async () => {
      const video = videoRef.current;
      if (!video || video.readyState < 2) return;

      canvas.width  = FRAME_SIZE;
      canvas.height = FRAME_SIZE;
      ctx.drawImage(video, 0, 0, FRAME_SIZE, FRAME_SIZE);

      const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
      const b64     = dataUrl.split(",")[1];

      try {
        await fetch(`${apiBase}/api/stream-frame`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, frame_b64: b64 }),
        });
      } catch (_) {
        // best-effort — silently skip on network hiccup
      }
    }, FRAME_INTERVAL_MS);

    return () => clearInterval(intervalRef.current);
  }, [cameraOn, sessionId, apiBase]);

  return (
    <div className="camera-feed">
      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} style={{ display: "none" }} />

      {/* Video preview */}
      <div className="camera-viewport">
        {cameraOn ? (
          <video ref={videoRef} className="camera-video" muted playsInline autoPlay />
        ) : (
          <div className="camera-placeholder">
            {error
              ? <p className="camera-error">{error}</p>
              : <p className="camera-hint">Camera is off</p>
            }
          </div>
        )}

        {/* Scene overlay */}
        {cameraOn && sceneInfo && (
          <div className="scene-overlay">
            <span className={`scene-badge ${sceneInfo.motion ? "motion" : ""}`}>
              {sceneInfo.people > 0 ? `👤 ${sceneInfo.people}` : "No people"}
              {sceneInfo.motion ? " · motion" : ""}
            </span>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="camera-controls">
        {!cameraOn ? (
          <button className="btn-primary" onClick={startCamera}>
            Start Camera
          </button>
        ) : (
          <button className="btn-secondary" onClick={stopCamera}>
            Stop Camera
          </button>
        )}
      </div>
    </div>
  );
}
