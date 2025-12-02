import os
from dotenv import load_dotenv
import torch
import sounddevice as sd
import numpy as numpy
import win32com.client
import soundfile as sf
import scipy.io.wavfile as wav
from transformers import pipeline
import google.generativeai as genai
import time

# Configure the GEMINI API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

# Check if key loaded correctly
if not GOOGLE_API_KEY:
    print(
        "❌ ERROR: API Key not found. Please create a .env file or paste the key directly."
    )

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# Define the folder where you pasted ffmpeg.exe
# We use r"" to make sure Windows backslashes don't cause errors
ffmpeg_folder = r"E:\learning-python\MLflow\speech-recognition-app"

# Add this folder to the Windows PATH variable
os.environ["PATH"] += os.pathsep + ffmpeg_folder
# Setup TTS Engine using Windows SAPI (more reliable than pyttsx3)
speaker = win32com.client.Dispatch("SAPI.SpVoice")
speaker.Rate = 1  # Speed: -10 to 10, default is 0
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading Whisper Turbo on {device} and torch_dtype {dtype}")
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    device=device,
    dtype=dtype,
)

# CONFIG FOR RECORDING

sample_rate = 16000
threshold = 600
silence_limit = 2  # seconds
temp_file = "temp.wav"


def speak(text):
    """Tells the computer to speak the text aloud using Windows SAPI"""
    try:
        speaker.Speak(text)
    except Exception as e:
        pass


def smart_record_audio():
    print("\n🔴Recording audio...", end="", flush=True)

    audio_buffer = []  # Stores valid audio data
    silence_start = None
    started_speaking = False

    # We open a "Stream" to listen continuously
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16") as stream:
        while True:
            chunk, _ = stream.read(int(sample_rate * 0.1))

            volume = numpy.linalg.norm(chunk) / numpy.sqrt(len(chunk))

            if volume > threshold:
                if not started_speaking:
                    print("\n🔴 Speech Detected! Recording...", end="", flush=True)
                    started_speaking = True

                # Add audio to buffer
                audio_buffer.extend(chunk)
                silence_start = None  # Reset silence start

            elif started_speaking:

                audio_buffer.extend(chunk)

                if silence_start is None:
                    silence_start = time.time()

                # Check if we have been silent for too long

                if time.time() - silence_start > silence_limit:
                    started_speaking = False
                    break  # Stop recording

            else:
                pass  # We haven't started speaking yet, just ignore the noise

    # Save the audio to a temporary file
    audio_numpy = numpy.array(audio_buffer, dtype="int16")
    sf.write(temp_file, audio_numpy, sample_rate)
    return True


# MAIN LOOP

speak("Smart Listening Mode Online.")

try:
    while True:
        # This function now BLOCKS until you finish a sentence
        smart_record_audio()

        # Transcribe audio we trimmed silence hallucinations  should be gone.

        result = pipe(temp_file)
        user_text = result["text"]

        if user_text.strip() != "":
            print(f"You: {user_text}")

            # The Logic for AI model
            # This is where you would normally send 'user_text' to Gemini/OpenAI.
            # For now, we use simple Python logic:

            if "stop" in user_text.lower() or "exit" in user_text.lower():
                speak("Goodbye! Have a great day!")
                break

            try:
                # Ask Gemini for response

                response = model.generate_content(
                    user_text + "(Answer in 1 short sentence)"
                )
                ai_reply = response.text
                print(f"🤖 AI: {ai_reply}")
                speak(ai_reply)
            except Exception as e:
                print(f"Gemini error: {e}")
                speak("I'm sorry, I didn't understand that. Please try again.")

except KeyboardInterrupt:
    print("\nTranscription stopped by user.")
