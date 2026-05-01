# import os
# import elevenlabs
# from elevenlabs.client import ElevenLabs
# import subprocess
# import platform
# from dotenv import load_dotenv

# load_dotenv()

# ELEVENLABS_API_KEY=os.getenv("ELEVENLABS_API_KEY")


# def text_to_speech_with_elevenlabs(input_text, output_filepath):
#     if not ELEVENLABS_API_KEY:
#         raise ValueError("ELEVENLABS_API_KEY missing from .env!")
#     client=ElevenLabs(api_key=ELEVENLABS_API_KEY)
#     audio=client.text_to_speech.convert(
#         text= input_text,
#         voice_id="21m00Tcm4TlvDq8ikWAM", #"JBFqnCBsd6RMkjVDRZzb",
#         model_id="eleven_multilingual_v2",
#         output_format= "mp3_22050_32",
#     )
#     elevenlabs.save(audio, output_filepath)
#     os_name = platform.system()
#     try:
#         if os_name == "Darwin":  # macOS
#             subprocess.run(['afplay', output_filepath])
#         elif os_name == "Windows":  # Windows
#             subprocess.run(['ffplay', '-nodisp', '-autoexit', output_filepath],
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.DEVNULL
#             )
#         elif os_name == "Linux":  # Linux
#             subprocess.run(['aplay', output_filepath])  # Alternative: use 'mpg123' or 'ffplay'
#         else:
#             raise OSError("Unsupported operating system")
#     except Exception as e:
#         print(f"An error occurred while trying to play the audio: {e}")


# from gtts import gTTS

# def text_to_speech_with_gtts(input_text, output_filepath):
#     language="en"

#     audioobj= gTTS(
#         text=input_text,
#         lang=language,
#         slow=False
#     )
#     audioobj.save(output_filepath)
#     os_name = platform.system()
#     try:
#         if os_name == "Darwin":  # macOS
#             subprocess.run(['afplay', output_filepath])
#         elif os_name == "Windows":  # Windows
#             subprocess.run(['ffplay', '-nodisp', '-autoexit', output_filepath],
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.DEVNULL
#             )
#         elif os_name == "Linux":  # Linux
#             subprocess.run(['aplay', output_filepath])  # Alternative: use 'mpg123' or 'ffplay'
#         else:
#             raise OSError("Unsupported operating system")
#     except Exception as e:
#         print(f"An error occurred while trying to play the audio: {e}")


# # input_text = "Hi, I am doing fine, how are you? This is a test for AI with Vinoothna"
# # output_filepath = "test_text_to_speech.mp3"
# # text_to_speech_with_elevenlabs(input_text, output_filepath)
# # #text_to_speech_with_gtts(input_text, output_filepath)

"""
text_to_speech.py
-----------------
Text-to-speech using gTTS (primary) or ElevenLabs (optional).
Audio playback runs in a background thread so it never blocks the UI.
"""

import os
import subprocess
import platform
import threading
import logging
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ── Internal: non-blocking playback ───────────────────────────────────────────

def _play_audio_nonblocking(filepath: str) -> None:
    """Play an audio file in a background thread (non-blocking)."""
    def _play():
        os_name = platform.system()
        try:
            if os_name == "Darwin":
                subprocess.run(["afplay", filepath], check=True)
            elif os_name == "Windows":
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", filepath],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
            elif os_name == "Linux":
                subprocess.run(["mpg123", "-q", filepath], check=True)
        except FileNotFoundError:
            logger.warning("Audio player not found for OS: %s. Skipping playback.", os_name)
        except subprocess.CalledProcessError as e:
            logger.error("Audio playback error: %s", e)

    thread = threading.Thread(target=_play, daemon=True, name="AudioPlayback")
    thread.start()


# ── Public API ─────────────────────────────────────────────────────────────────

def text_to_speech_with_gtts(input_text: str, output_filepath: str = "voice.mp3") -> None:
    """
    Convert text to speech using gTTS and play it asynchronously.

    Args:
        input_text: The text to speak.
        output_filepath: Where to save the mp3 file.
    """
    if not input_text.strip():
        return

    try:
        tts = gTTS(text=input_text, lang="en", slow=False)
        tts.save(output_filepath)
        _play_audio_nonblocking(output_filepath)
    except Exception as e:
        logger.error("gTTS failed: %s", e)


def text_to_speech_with_elevenlabs(input_text: str, output_filepath: str = "voice.mp3") -> None:
    """
    Convert text to speech using ElevenLabs (optional premium quality).
    Requires ELEVENLABS_API_KEY in .env
    """
    try:
        import elevenlabs
        from elevenlabs.client import ElevenLabs

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not set.")

        client = ElevenLabs(api_key=api_key)
        audio = client.text_to_speech.convert(
            text=input_text,
            voice_id="21m00Tcm4TlvDq8ikWAM",
            model_id="eleven_multilingual_v2",
            output_format="mp3_22050_32",
        )
        elevenlabs.save(audio, output_filepath)
        _play_audio_nonblocking(output_filepath)

    except Exception as e:
        logger.error("ElevenLabs TTS failed: %s", e)