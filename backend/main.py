from __future__ import annotations
import asyncio, base64, logging, os, tempfile, uuid
from contextlib import asynccontextmanager
from pathlib import Path
import cv2, uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_server_camera_active = False
_session_frames: dict[str, str] = {}

def _try_start_server_camera():
    global _server_camera_active
    try:
        from services.camera_stream import camera_stream
        from services.perception_loop import start_perception_loop
        camera_stream.start()
        start_perception_loop()
        _server_camera_active = True
        logger.info("Server-side camera started.")
    except Exception as e:
        logger.info("Server-side camera not available (%s)", e)

@asynccontextmanager
async def lifespan(_: FastAPI):
    _try_start_server_camera()
    yield
    if _server_camera_active:
        try:
            from services.camera_stream import camera_stream
            camera_stream.stop()
        except Exception:
            pass

app = FastAPI(title="Vision Assistant API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health")
async def health():
    return {"status": "ok", "server_camera": _server_camera_active}

@app.get("/api/camera-feed")
async def camera_feed():
    try:
        from services.camera_stream import camera_stream
        frame = camera_stream.get_frame()
        if frame is None:
            return JSONResponse({"frame": None})
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buf).decode()
        return JSONResponse({"frame": b64})
    except Exception:
        return JSONResponse({"frame": None})

@app.get("/api/scene")
async def get_scene():
    try:
        from services.scene_memory import get_scene
        return get_scene()
    except Exception:
        return {"people": 0, "motion": False, "last_update": None}

class FramePayload(BaseModel):
    session_id: str
    frame_b64: str

@app.post("/api/stream-frame")
async def receive_frame(payload: FramePayload):
    _session_frames[payload.session_id] = payload.frame_b64
    return {"ok": True}

@app.post("/api/transcribe")
async def transcribe(file: UploadFile):
    if not file.filename:
        raise HTTPException(400, "No file provided")
    tmp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}.webm"
    try:
        tmp_path.write_bytes(await file.read())
        from services.speech_to_text import transcribe_with_groq
        text = transcribe_with_groq(str(tmp_path))
        return {"text": text}
    finally:
        tmp_path.unlink(missing_ok=True)
@app.post("/api/tts")
async def text_to_speech_api(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return JSONResponse({"error": "No text"}, status_code=400)
    try:
        from gtts import gTTS
        import io
        # Detect language
        import re
        if re.search(r'[\u0C00-\u0C7F]', text):
            lang = "te"
        elif re.search(r'[\u0900-\u097F]', text):
            lang = "hi"
        elif re.search(r'[\u0B80-\u0BFF]', text):
            lang = "ta"
        else:
            lang = "en"
        tts = gTTS(text=text, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        from fastapi.responses import StreamingResponse
        return StreamingResponse(buf, media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
@app.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info("WS connected: %s", session_id)
    from agent.graph import drop_session, get_session
    from langchain_core.messages import HumanMessage, AIMessage
    import agent.tools as tool_module

    session = get_session(session_id)
    graph = session["graph"]

    async def keepalive():
        while True:
            await asyncio.sleep(30)
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

    keepalive_task = asyncio.create_task(keepalive())
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "pong":
                continue
            if data.get("type") != "message":
                continue
            user_text = data.get("text", "").strip()
            if not user_text:
                continue
            await websocket.send_json({"type": "start"})
            current_frame = _session_frames.get(session_id)
            tok = tool_module._current_frame_b64.set(current_frame)
            try:
                from agent.intent_router import classify as classify_intent
                intent = classify_intent(user_text)
                logger.info("Intent classified: %s", intent)

                # Trim history to last 6 messages, but never start on a ToolMessage
                # or orphaned AIMessage(tool_calls) — scan back to find a safe HumanMessage boundary
                msgs = session["messages"]
                if len(msgs) > 6:
                    trimmed = msgs[-6:]
                    # Walk forward until we start on a HumanMessage
                    while trimmed and not isinstance(trimmed[0], HumanMessage):
                        trimmed = trimmed[1:]
                    session["messages"] = trimmed if trimmed else msgs[-2:]

                state_input = {
                    "messages": session["messages"] + [HumanMessage(content=user_text)],
                    "current_frame_b64": current_frame,
                    "intent": intent,
                    "tool_call_count": 0,
                }
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: graph.invoke(state_input)
                )
                session["messages"] = list(result["messages"])
                last = result["messages"][-1]
                if isinstance(last, AIMessage) and last.content:
                    await websocket.send_json({"type": "token", "text": last.content})
                await websocket.send_json({"type": "end"})
            except Exception as e:
                logger.error("Agent error: %s", e)
                await websocket.send_json({"type": "error", "text": "Something went wrong. Please try again."})
            finally:
                tool_module._current_frame_b64.reset(tok)
    except WebSocketDisconnect:
        logger.info("WS disconnected: %s", session_id)
    except Exception as e:
        logger.error("WS error (%s): %s", session_id, e)
    finally:
        keepalive_task.cancel()
        drop_session(session_id)

FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")
    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        return FileResponse(str(FRONTEND_DIST / "index.html"))
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "Backend running. Start frontend: cd frontend && npm run dev"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
