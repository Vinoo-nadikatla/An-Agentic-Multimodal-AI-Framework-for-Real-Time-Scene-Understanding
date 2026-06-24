# An Agentic Multimodal AI Framework for Real-Time Scene Understanding Across Assistive and Industrial Safety Domains

> **MTech Project** — Visvesvaraya National Institute of Technology (VNIT) Nagpur

---

## Overview

This repository presents a unified agentic multimodal AI architecture that integrates real-time computer vision, large language model reasoning, multilingual speech processing, and automated safety monitoring into a single deployable system on commodity hardware — no GPU required.

The project evolved across three phases:

- **Phase 1 — Gradio Prototype** (`main` branch) — Initial proof-of-concept using Gradio UI with basic conversational AI and camera integration
- **Phase 2 — General Conversation Assistant** (`phase-2-webapp` branch) — Full React + FastAPI + LangGraph architecture with multilingual voice, scene memory, vision tool, and OCR
- **Phase 3 — Industrial Safety Monitor** (`industrial-safety` branch) — Extends Phase 2 with YOLOv8n PPE detection, dual-thread MJPEG streaming, worker tracking, compliance dashboard, and automated safety alerts
- **Additional Domains Analysed:** ATM surveillance assistant, visual assistance for persons with visual impairments
---

## Key Features

### Base Architecture (Phase 2 and Phase 3)

- **Zero-token intent router** — regex-based classification in under 10ms, zero API token cost
- **LangGraph agentic orchestration** — stateful agent graph with tool calling, memory injection, and conditional routing
- **Live webcam integration** — real-time scene understanding via Llama-4-Scout-17b vision-language model
- **Multilingual interaction** — English, Telugu, and Hindi via Unicode-based language detection
- **Voice input** — Groq-hosted Whisper large-v3-turbo for multilingual speech recognition
- **Text-to-speech** — Browser Speech API (English) and gTTS (Telugu, Hindi)
- **OCR capability** — EasyOCR for text extraction from live camera frames
- **Conversational memory** — 6-message sliding window history management
- **WebSocket communication** — persistent real-time bidirectional chat

### Phase 3 — Industrial Safety Extensions

- **YOLOv8n helmet detection** — Hardhat / NO-Hardhat classification, confidence threshold 0.40
- **Dual-thread MJPEG pipeline** — Thread 1 displays at 20fps; Thread 2 runs YOLO every 5th frame without blocking video
- **Centroid-based worker tracking** — session-wide Worker A / B / C identity assignment
- **Violation duration tracking** — per-worker non-compliance timer in MM:SS format
- **Automated TTS announcements** — fires after 30-second continuous violation
- **PPE compliance dashboard** — Overall%, Helmet%, Workers X/Y updated in real time
- **200-entry activity log** — ring buffer with timestamped safety events
- **5-section safety report** — generated on demand from activity log
- **Safety alert banner** — visual indicator on frontend dashboard

---

## System Architecture

```
User (text / voice)
        ↓
[Whisper STT] ──► WebSocket ──► FastAPI Backend
                                        ↓
                             Intent Router  (regex · <10ms · 0 tokens)
                             ┌──────┬──────┬──────┬────────┬─────────┐
                           scene  vision  ocr  report  general
                             ↓      ↓      ↓     ↓        ↓
                          ┌─────────────────────────────────────┐
                          │         LangGraph Agent              │
                          │  conversation_node                   │
                          │  tool_router → tool_executor         │
                          │  language_detection_node             │
                          │  synthesis_node                      │
                          └─────────────────────────────────────┘
                                        ↓
                             Llama-3.3-70b generates response
                                        ↓
                             WebSocket ──► React Frontend ──► TTS


Phase 3 Camera Pipeline (runs in parallel):

Webcam (OpenCV)
      ↓
Thread 1 — Camera Reader (20fps)
      ├── Every frame  → _cached_annotated_bytes → MJPEG stream → Browser
      └── Every 5th frame → Detection queue
                                ↓
Thread 2 — YOLO Detection
      ├── YOLOv8n inference → Hardhat / NO-Hardhat
      ├── Centroid tracker → Worker A / B / C
      ├── Draw bounding boxes (green = OK · red = violation)
      ├── Update PPE status cache + activity log
      └── Violation > 30s → Auto TTS announcement
```

---

## Tech Stack

| Category | Technology | Version / Variant |
|---|---|---|
| LLM Reasoning | Llama-3.3-70b (Groq API) | llama-3.3-70b-versatile |
| Vision-Language | Llama-4-Scout-17b (Groq API) | llama-4-scout-17b-16e-instruct |
| Speech-to-Text | Whisper large-v3-turbo (Groq API) | whisper-large-v3-turbo |
| Object Detection | YOLOv8n — Hard Hat (HuggingFace) | Ultralytics 8.4.48 |
| Agentic Framework | LangGraph | 0.2+ |
| Backend | FastAPI + Uvicorn | 0.115+ |
| Frontend | React + Vite | React 18 / Vite 5+ |
| Camera Capture | OpenCV VideoCapture | 4.10+ |
| Video Streaming | MJPEG | multipart/x-mixed-replace |
| OCR | EasyOCR | 1.7+ |
| TTS — English | Browser Speech API | Web standard |
| TTS — Telugu / Hindi | gTTS | 2.5+ |
| Object Tracking | Centroid Tracker | Custom implementation |
| Package Manager | uv (Python) | 0.5+ |
| Version Control | Git / GitHub | — |

> All LLM and vision model inference is offloaded to the Groq Cloud API.
> No dedicated GPU is required. The complete system runs on a standard laptop CPU.

---

## Repository Structure

```
├── backend/
│   ├── agent/
│   │   ├── graph.py             # LangGraph agent graph definition
│   │   ├── intent_router.py     # Zero-token regex intent classifier
│   │   ├── nodes.py             # LangGraph conversation and tool nodes
│   │   └── tools.py             # Vision tool, OCR tool
│   ├── services/
│   │   ├── camera_stream.py     # OpenCV webcam capture
│   │   ├── perception_loop.py   # Dual-thread pipeline (Phase 3)
│   │   ├── ppe_detector.py      # YOLOv8n + centroid tracker (Phase 3)
│   │   ├── scene_memory.py      # Scene state in-memory cache
│   │   ├── activity_log.py      # 200-entry ring buffer (Phase 3)
│   │   ├── speech_to_text.py    # Whisper STT via Groq
│   │   └── text_to_speech.py    # gTTS for Telugu / Hindi
│   ├── main.py                  # FastAPI app + WebSocket + API endpoints
│   └── .env                     # API keys — not committed to git
├── frontend/
│   ├── src/
│   │   └── App.jsx              # React UI — chat + camera + dashboard
│   └── package.json
└── pyproject.toml
```

---

## Setup and Installation

### Prerequisites

- Python 3.12+
- Node.js 18+
- uv Python package manager — https://docs.astral.sh/uv/
- Groq API key — free at https://console.groq.com
- Webcam (built-in or USB)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Vinoo-nadikatla/An-Agentic-Multimodal-AI-Framework-for-Real-Time-Scene-Understanding.git
cd An-Agentic-Multimodal-AI-Framework-for-Real-Time-Scene-Understanding
```

### Step 2 — Create .env File

```bash
# Windows
echo GROQ_API_KEY=your_groq_api_key_here > backend\.env

# Mac / Linux
echo "GROQ_API_KEY=your_groq_api_key_here" > backend/.env
```

### Step 3 — Run Phase 2 (General Conversation Assistant)

```bash
git checkout phase-2-webapp

# Terminal 1 — Backend
uv run backend/main.py

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
```

Open browser: **http://localhost:5173**

### Step 4 — Run Phase 3 (Industrial Safety Monitor)

```bash
git checkout industrial-safety

# Terminal 1 — Backend
uv run backend/main.py

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
```

Open browser: **http://localhost:5173**

> Note: Phase 2 and Phase 3 cannot run simultaneously as both use port 8000.

---

## Demo Queries

### Phase 2 — General Conversation Assistant

| Query | Intent | Description |
|---|---|---|
| How many people are in the room? | Scene | Returns count from cache — no API call |
| What do you see right now? | Vision | Live camera description via Llama-4-Scout |
| Describe my surroundings | Vision | Full environment analysis |
| Is there any movement? | Scene | Motion detection state |
| Read the text on this label | OCR | EasyOCR text extraction |
| What is artificial intelligence? | General | LLM knowledge response |
| ఎంత మంది కార్మికులు ఉన్నారు? | Scene (Telugu) | Response in Telugu |
| कितने लोग हैं? | Scene (Hindi) | Response in Hindi |

### Phase 3 — Industrial Safety Monitor

| Query | Intent | Description |
|---|---|---|
| How many workers are present? | Scene | Count from PPE cache |
| Is the worker wearing a helmet? | Vision | YOLOv8 + visual analysis |
| What is the current PPE compliance? | Scene | Compliance % from cache |
| Give me a safety report | Report | 5-section structured report |
| Is anyone missing a helmet? | Vision | Live camera detection |
| Check PPE status | Scene | Per-worker compliance status |
| కార్మికులు హెల్మెట్ ధరించారా? | Vision (Telugu) | Telugu safety response |
| कितने मजदूर हैं? | Scene (Hindi) | Hindi worker count |

---

## Performance Results

| Metric | Result |
|---|---|
| Intent classification accuracy | 98.3% (59 / 60 test queries) |
| Intent routing latency | <10ms (zero LLM tokens consumed) |
| Scene query latency (cached) | ~380ms ±48ms |
| General knowledge latency | ~664ms ±154ms |
| Vision describe latency | ~2,509ms ±405ms (3 LLM calls) |
| Safety report latency | ~1,890ms ±210ms |
| Helmet detection F1 score | 0.88 (frontal average) |
| Helmet detection Precision | 0.88 |
| Helmet detection Recall | 0.87 |
| Video frame rate (dual-thread) | 18–20fps display |
| Detection overlay update rate | ~4fps (every 5th frame) |
| RAM usage | ~1.8GB |
| GPU required | None — CPU only |

---

## Branches


| Branch | Description | UI Framework |
|---|---|---|
| `main` | Phase 1 — Gradio prototype, basic conversational AI + camera | Gradio |
| `phase-2-webapp` | Phase 2 — General Conversation Assistant with multilingual voice, vision, OCR | React + Vite |
| `industrial-safety` | Phase 3 — Industrial Safety Monitor with PPE detection and compliance dashboard | React + Vite |

---

## Application Domains

| Domain | Key Capabilities |
|---|---|
| General Conversation Assistant | Scene memory, live vision, OCR, multilingual, voice |
| Industrial Safety Monitor | PPE detection, compliance dashboard, auto-alerts, safety reports |
| ATM Surveillance | Security monitoring, person detection, dwell-time alerts |
| Blind Person Assistant | Voice-first interaction, spatial description, product label OCR |

All domains reuse the same base architecture. Only the system prompt, intent patterns, and optional domain services differ.

---

## Comparison with Related Work

| Capability | Fang et al. | Nath et al. | Wu et al. | This Work |
|---|---|---|---|---|
| PPE Detection | Helmet | Multi-class | Multi-class | Helmet (extensible) |
| Hardware | GPU server | GPU server | Nvidia Jetson | Laptop CPU |
| Conversational Interface | None | None | None | Yes — 3 languages |
| Voice Interaction | None | None | None | EN voice + TE/HI text |
| Assistive Application | None | None | None | Blind assistant, ATM |
| Temporal Queries | None | None | None | Yes — activity log |
| Safety Reports | None | None | None | Yes — 5 sections |
| Automated Alerts | None | None | Dashboard only | TTS + visual |
| Open-weight Models | Partial | Partial | Partial | All open-weight |
| Agentic Reasoning | None | None | None | LangGraph agent |

---


## Future Work

- Multi-class PPE detection (safety vests, gloves, footwear, eye protection)
- Multi-camera monitoring with cross-camera worker re-identification
- Improved Indic language ASR using AI4Bharat IndicASR
- Persistent storage for cross-session compliance analytics
- Edge deployment with locally hosted models (no internet dependency)
- Appearance-based worker re-identification (OSNet / DeepSORT)
- Enterprise safety management system integration

---

## Citation

If you use this work, please cite:

```
Nadikatla, V. (2026). An Agentic Multimodal AI Framework for Real-Time Scene 
Understanding Across Assistive and Industrial Safety Domains.
MTech Dissertation, Department of ECE - Applied Artificial Intelligence,
Visvesvaraya National Institute of Technology, Nagpur.
```

---

## License

This project was developed as part of an MTech dissertation at VNIT Nagpur.
© 2026 Vinoothna Nadikatla, VNIT Nagpur. All rights reserved.
