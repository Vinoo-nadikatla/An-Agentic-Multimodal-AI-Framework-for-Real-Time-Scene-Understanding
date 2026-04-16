# # import os
# # import gradio as gr
# # import cv2
# # from dotenv import load_dotenv
# # from camera_stream import camera_manager

# # # Local modules
# # from speech_to_text import record_audio, transcribe_with_groq
# # from ai_agent import ask_agent
# # from text_to_speech import text_to_speech_with_elevenlabs

# # load_dotenv()

# # # -------------------------
# # # GLOBALS
# # # -------------------------
# # audio_filepath = "audio_question.mp3"

# # LISTENING = False
# # camera = None
# # streaming = False
# # last_frame = None


# # # -------------------------
# # # LISTENING LOGIC
# # # -------------------------
# # def toggle_listening():
# #     global LISTENING
# #     LISTENING = not LISTENING
# #     return "🟢 Listening Enabled" if LISTENING else "🔴 Listening Disabled"


# # def listen_and_process(chat_history):
# #     """Runs periodically when listening mode is ON."""
# #     global LISTENING

# #     if not LISTENING:
# #         return chat_history

# #     # Record voice
# #     record_audio(file_path=audio_filepath)

# #     # Speech → Text
# #     user_text = transcribe_with_groq(audio_filepath)

# #     if not user_text.strip():
# #         return chat_history

# #     # AI response
# #     bot_reply = ask_agent(user_query=user_text)

# #     # Text → Speech
# #     text_to_speech_with_elevenlabs(bot_reply, "final.mp3")

# #     # Update chat
# #     chat_history.append({"role": "user", "content": user_text})
# #     chat_history.append({"role": "assistant", "content": bot_reply})

# #     return chat_history


# # # -------------------------
# # # TEXT CHAT
# # # -------------------------
# # def text_chat(user_message, chat_history):
# #     if not user_message:
# #         return "", chat_history

# #     bot_reply = ask_agent(user_query=user_message)

# #     chat_history.append({"role": "user", "content": user_message})
# #     chat_history.append({"role": "assistant", "content": bot_reply})

# #     text_to_speech_with_elevenlabs(bot_reply, "final.mp3")

# #     return "", chat_history


# # # -------------------------
# # # CAMERA
# # # -------------------------
# # def start_camera():
# #     global camera, streaming

# #     camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# #     streaming = True

# #     return "Camera Started"


# # def stop_camera():
# #     global camera, streaming

# #     streaming = False

# #     if camera is not None:
# #         camera.release()

# #     return None


# # def get_frame():
# #     global camera, streaming, last_frame

# #     if not streaming or camera is None:
# #         return last_frame

# #     ret, frame = camera.read()

# #     if ret:
# #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         last_frame = frame
# #         return frame

# #     return last_frame


# # # -------------------------
# # # PROFESSIONAL CSS
# # # -------------------------
# # css = """
# # body{
# # background: linear-gradient(135deg,#0f172a,#1e293b);
# # font-family: 'Inter', sans-serif;
# # }

# # #title{
# # text-align:center;
# # font-size:32px;
# # font-weight:700;
# # color:white;
# # margin-bottom:8px;
# # }

# # #subtitle{
# # text-align:center;
# # font-size:15px;
# # color:#cbd5f5;
# # margin-bottom:25px;
# # }

# # .panel{
# # background:#111827;
# # border-radius:14px;
# # padding:15px;
# # box-shadow:0px 5px 25px rgba(0,0,0,0.4);
# # }

# # button{
# # border-radius:10px !important;
# # }
# # """


# # # ============================
# # # INTERFACE
# # # ============================
# # with gr.Blocks(css=css) as demo:

# #     gr.Markdown("<div id='title'>VN Agentic Multimodal AI Assistant</div>")
# #     gr.Markdown("<div id='subtitle'>Voice • Vision • Intelligence</div>")

# #     with gr.Row():

# #         # -------------------------
# #         # CAMERA PANEL
# #         # -------------------------
# #         with gr.Column(scale=1, elem_classes="panel"):

# #             gr.Markdown("### 📷 Live Camera")

# #             webcam_output = gr.Image(height=420)

# #             with gr.Row():
# #                 start_btn = gr.Button("Start Camera", variant="primary")
# #                 stop_btn = gr.Button("Stop Camera")

# #             cam_timer = gr.Timer(0.05)

# #         # -------------------------
# #         # CHAT PANEL
# #         # -------------------------
# #         with gr.Column(scale=1.2, elem_classes="panel"):

# #             gr.Markdown("### 💬 AI Conversation")

# #             chatbot = gr.Chatbot(
# #                 height=420,
# #                 type="messages",
# #                 bubble_full_width=False
# #             )

# #             with gr.Row():
# #                 text_input = gr.Textbox(
# #                     placeholder="Ask anything...",
# #                     show_label=False,
# #                     scale=8
# #                 )

# #                 send_btn = gr.Button("Send", variant="primary", scale=1)

# #             with gr.Row():
# #                 listen_btn = gr.Button("🎤 Voice Mode", variant="primary")
# #                 clear_btn = gr.Button("🧹 Clear Chat")

# #             listen_status = gr.Markdown("🔴 Listening Disabled")


# #     # -------------------------
# #     # EVENTS
# #     # -------------------------
# #     start_btn.click(start_camera)
# #     stop_btn.click(stop_camera)

# #     cam_timer.tick(get_frame, outputs=webcam_output)

# #     listen_btn.click(toggle_listening, outputs=listen_status)

# #     cam_timer.tick(
# #         listen_and_process,
# #         inputs=chatbot,
# #         outputs=chatbot,
# #         show_progress=False
# #     )

# #     text_input.submit(
# #         text_chat,
# #         inputs=[text_input, chatbot],
# #         outputs=[text_input, chatbot]
# #     )

# #     send_btn.click(
# #         text_chat,
# #         inputs=[text_input, chatbot],
# #         outputs=[text_input, chatbot]
# #     )

# #     clear_btn.click(lambda: [], outputs=chatbot)


# # # -------------------------
# # # LAUNCH
# # # -------------------------
# # if __name__ == "__main__":

# #     demo.launch(
# #         server_name="127.0.0.1",
# #         server_port=7860,
# #         share=False,
# #         debug=True
# #     )

# #////////////////////////////////////
# # from camera_stream import camera_stream
# # from perception_loop import start_perception
# # from ai_agent import ask_agent

# # camera_stream.start()

# # start_perception()

# # while True:

# #     query=input("Ask: ")

# #     response=ask_agent(query)

# #     print(response)

# import gradio as gr
# import cv2
# from dotenv import load_dotenv

# from camera_stream import camera_stream
# from perception_loop import start_perception
# from scene_memory import get_scene
# from ai_agent import ask_agent

# from speech_to_text import transcribe_with_groq
# from text_to_speech import text_to_speech_with_gtts

# load_dotenv()

# camera_stream.start()
# start_perception()


# # -------------------------
# # CAMERA FRAME
# # -------------------------

# def get_frame():

#     frame = camera_stream.get_frame()

#     if frame is None:
#         return None

#     return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# # -------------------------
# # SCENE STATUS
# # -------------------------

# def scene_status():

#     scene = get_scene()

#     return f"""
# ### Scene Intelligence

# 👥 People detected: **{scene['people']}**

# 🎯 Motion detected: **{scene['motion']}**
# """


# # -------------------------
# # TEXT CHAT
# # -------------------------

# def chat_text(message, history):

#     if not message:
#         return "", history

#     response = ask_agent(message)

#     history.append({"role":"user","content":message})
#     history.append({"role":"assistant","content":response})

#     text_to_speech_with_gtts(response,"voice.mp3")

#     return "", history


# # -------------------------
# # VOICE CHAT
# # -------------------------

# def chat_voice(audio, history):

#     if audio is None:
#         return history

#     text = transcribe_with_groq(audio)

#     response = ask_agent(text)

#     history.append({"role":"user","content":text})
#     history.append({"role":"assistant","content":response})

#     text_to_speech_with_gtts(response,"voice.mp3")

#     return history


# # -------------------------
# # CSS
# # -------------------------

# css = """
# body{
# background: linear-gradient(135deg,#0f172a,#1e293b);
# font-family: 'Inter', sans-serif;
# color:white;
# }

# #header{
# text-align:center;
# font-size:36px;
# font-weight:700;
# margin-bottom:10px;
# }

# #subheader{
# text-align:center;
# font-size:16px;
# color:#94a3b8;
# margin-bottom:25px;
# }

# .panel{
# background:#111827;
# border-radius:16px;
# padding:20px;
# box-shadow:0px 10px 40px rgba(0,0,0,0.5);
# }

# button{
# border-radius:10px !important;
# font-weight:600 !important;
# }

# .gr-chatbot{
# background:#020617 !important;
# }
# """


# # -------------------------
# # UI
# # -------------------------

# with gr.Blocks(css=css) as demo:

#     gr.Markdown("<div id='header'>VN Multimodal AI Assistant</div>")
#     gr.Markdown("<div id='subheader'>Vision • Voice • Reasoning</div>")

#     with gr.Row():

#         # LEFT PANEL
#         with gr.Column(scale=1, elem_classes="panel"):

#             gr.Markdown("### 📷 Live Camera")

#             webcam = gr.Image(height=420)

#             scene_info = gr.Markdown()

#             cam_timer = gr.Timer(0.1)


#         # RIGHT PANEL
#         with gr.Column(scale=1, elem_classes="panel"):

#             gr.Markdown("### 💬 AI Conversation")

#             chatbot = gr.Chatbot(height=420, type="messages")

#             text_input = gr.Textbox(
#                 placeholder="Ask anything...",
#                 show_label=False
#             )

#             send_btn = gr.Button("Send", variant="primary")

#             voice_input = gr.Audio(
#                 sources=["microphone"],
#                 type="filepath"
#             )

#             clear_btn = gr.Button("Clear Chat")


#     # -------------------------
#     # EVENTS
#     # -------------------------

#     cam_timer.tick(get_frame, outputs=webcam)

#     cam_timer.tick(scene_status, outputs=scene_info)

#     text_input.submit(
#         chat_text,
#         inputs=[text_input, chatbot],
#         outputs=[text_input, chatbot]
#     )

#     send_btn.click(
#         chat_text,
#         inputs=[text_input, chatbot],
#         outputs=[text_input, chatbot]
#     )

#     voice_input.stop_recording(
#         chat_voice,
#         inputs=[voice_input, chatbot],
#         outputs=chatbot
#     )

#     clear_btn.click(lambda: [], outputs=chatbot)


# demo.launch(
#     server_name="127.0.0.1",
#     server_port=7860,
#     share=False,
#     debug=True
# )

"""
main.py
-------
Gradio UI for the Multimodal AI Assistant.

Key fix: cam_timer.tick() can only bind ONE callback per call.
To update both the webcam image and scene status, we use a single
callback that returns multiple outputs as a tuple.
"""

import cv2
import logging
import gradio as gr
from dotenv import load_dotenv

from camera_stream import camera_stream
from perception_loop import start_perception
from scene_memory import get_scene
from ai_agent import ask_agent
from speech_to_text import transcribe_with_groq
from text_to_speech import text_to_speech_with_gtts

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Startup ────────────────────────────────────────────────────────────────────

camera_stream.start()
start_perception()

# ── Callbacks ──────────────────────────────────────────────────────────────────

def get_camera_and_scene():
    """
    Single timer callback — returns (frame, scene_markdown) as a tuple.
    This is the fix for the double cam_timer.tick() bug: one callback,
    two outputs.
    """
    frame = camera_stream.get_frame()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None

    scene = get_scene()
    motion_str = "🟢 Yes" if scene["motion"] else "🔴 No"
    scene_md = (
        f"### Scene Intelligence\n\n"
        f"👥 **People detected:** {scene['people']}\n\n"
        f"🎯 **Motion detected:** {motion_str}"
    )

    return rgb_frame, scene_md


def chat_text(message: str, history: list) -> tuple:
    if not message.strip():
        return "", history

    response = ask_agent(message)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    # TTS runs in background — does not block the UI response
    text_to_speech_with_gtts(response, "voice.mp3")

    return "", history


def chat_voice(audio_filepath, history: list) -> list:
    if audio_filepath is None:
        return history

    text = transcribe_with_groq(audio_filepath)
    if not text:
        history.append({"role": "assistant", "content": "Sorry, I couldn't hear that clearly."})
        return history

    response = ask_agent(text)

    history.append({"role": "user", "content": f"🎤 {text}"})
    history.append({"role": "assistant", "content": response})

    text_to_speech_with_gtts(response, "voice.mp3")

    return history


# ── CSS ────────────────────────────────────────────────────────────────────────

css = """
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    font-family: 'Inter', sans-serif;
    color: white;
}
#header {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 10px;
}
#subheader {
    text-align: center;
    font-size: 16px;
    color: #94a3b8;
    margin-bottom: 25px;
}
.panel {
    background: #111827;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0px 10px 40px rgba(0,0,0,0.5);
}
button { border-radius: 10px !important; font-weight: 600 !important; }
"""

# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(css=css) as demo:

    gr.Markdown("<div id='header'>VN Multimodal AI Assistant</div>")
    gr.Markdown("<div id='subheader'>Vision • Voice • Reasoning</div>")

    with gr.Row():

        # LEFT: Camera + Scene
        with gr.Column(scale=1, elem_classes="panel"):
            gr.Markdown("### 📷 Live Camera")
            webcam = gr.Image(height=420, label=None, show_label=False)
            scene_info = gr.Markdown("Loading scene data…")
            cam_timer = gr.Timer(0.1)

        # RIGHT: Chat
        with gr.Column(scale=1, elem_classes="panel"):
            gr.Markdown("### 💬 AI Conversation")
            chatbot = gr.Chatbot(height=380, type="messages")
            text_input = gr.Textbox(placeholder="Ask anything…", show_label=False)
            send_btn = gr.Button("Send", variant="primary")
            voice_input = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Voice Input")
            clear_btn = gr.Button("Clear Chat")

    # ── Events ────────────────────────────────────────────────────────────────

    # FIX: single tick callback with two outputs (fixes double-bind bug)
    cam_timer.tick(
        fn=get_camera_and_scene,
        outputs=[webcam, scene_info],
    )

    text_input.submit(chat_text, inputs=[text_input, chatbot], outputs=[text_input, chatbot])
    send_btn.click(chat_text, inputs=[text_input, chatbot], outputs=[text_input, chatbot])
    voice_input.stop_recording(chat_voice, inputs=[voice_input, chatbot], outputs=chatbot)
    clear_btn.click(lambda: [], outputs=chatbot)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True,
    )
