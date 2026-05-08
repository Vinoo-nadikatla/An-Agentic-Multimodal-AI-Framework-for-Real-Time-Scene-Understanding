# import logging
# import speech_recognition as sr
# from pydub import AudioSegment
# from io import BytesIO
# import os
# from groq import Groq
# from dotenv import load_dotenv

# load_dotenv()  

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def record_audio(file_path, timeout=20, phrase_time_limit=None):
#     """
#     Function to record audio from the microphone and save it as an MP3 file.

#     Args:
#     file_path (str): Path to save the recorded audio file.
#     timeout (int): Maximum time to wait for a phrase to start (in seconds).
#     phrase_time_lfimit (int): Maximum time for the phrase to be recorded (in seconds).
#     """
#     recognizer = sr.Recognizer()
    
#     try:
#         with sr.Microphone() as source:
#             logging.info("Adjusting for ambient noise...")
#             recognizer.adjust_for_ambient_noise(source, duration=1)
#             logging.info("Start speaking now...")
            
#             # Record the audio
#             audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
#             logging.info("Recording complete.")
            
#             # Convert the recorded audio to an MP3 file
#             wav_data = audio_data.get_wav_data()
#             audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
#             audio_segment.export(file_path, format="mp3", bitrate="128k")
            
#             logging.info(f"Audio saved to {file_path}")

#     except Exception as e:
#         logging.error(f"An error occurred: {e}")

# # file_path = "test_speech_to_text.mp3"
# # record_audio(file_path)


# def transcribe_with_groq(audio_filepath):
#     GROQ_API_KEY=os.getenv("GROQ_API_KEY")
#     if not GROQ_API_KEY:
#         raise ValueError("❌ GROQ_API_KEY not found! Make sure it's in your .env file.")
#     client=Groq(api_key=GROQ_API_KEY)
#     stt_model="whisper-large-v3-turbo"
#     with open(audio_filepath, "rb") as audio_file:
#         transcription=client.audio.transcriptions.create(
#             model=stt_model,
#             file=audio_file,
#             language="en"
#         )

#     return transcription.text

# # audio_filepath = "test_speech_to_text.mp3"
# # print(transcribe_with_groq(audio_filepath))

"""
speech_to_text.py
-----------------
Transcribes audio files using Groq's Whisper API.
Gradio records audio and saves it as a filepath — we pass that directly to Groq.
"""

import os
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

STT_MODEL = "whisper-large-v3-turbo"


def transcribe_with_groq(audio_filepath: str) -> str:
    """
    Transcribe an audio file using Groq Whisper.

    Args:
        audio_filepath: Path to the audio file (mp3, wav, m4a, webm supported).

    Returns:
        Transcribed text string.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment.")

    client = Groq(api_key=api_key)

    try:
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=STT_MODEL,
                file=audio_file,
               
            )
        logger.info("Transcription: %s", transcription.text)
        return transcription.text

    except Exception as e:
        logger.error("Transcription failed: %s", e)
        return ""