import os
from dotenv import load_dotenv
import torch
import sounddevice as sd
import numpy as numpy
import win32com.client
import scipy.io.wavfile as wav
from transformers import pipeline
import google.generativeai as genai

# Configure the GEMINI API key
load_dotenv() 
GOOGLE_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

# Check if key loaded correctly
if not GOOGLE_API_KEY:
    print("❌ ERROR: API Key not found. Please create a .env file or paste the key directly.")

genai.configure(api_key = GOOGLE_API_KEY)
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
duration = 5  # seconds
temp_file = "temp.wav"


def speak(text):
    """Tells the computer to speak the text aloud using Windows SAPI"""
    try:
        speaker.Speak(text)
    except Exception as e:
        print(f"[DEBUG] Speech error: {e}")


def record_audio(duration):
    print("\n🔴Recording audio...") 
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        device=6,  # Explicitly use Microphone (Realtek HD Audio Mic input)
    )
    sd.wait()
    print("Processing audio...")

    wav.write(temp_file, sample_rate, audio_data)


# THE INFINITE LOOP

speak("System online.I am ready to help you.")

try:
    while True:
        # Record Audio
        record_audio(duration)

        # Transcribe Audio
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
            #Ask Gemini for response

            response = model.generate_content(user_text + "(Answer in 1 short sentence)")
            ai_reply = response.text
            print(f"🤖 AI: {ai_reply}")
            speak(ai_reply)
        except Exception as e:
            print(f"Gemini error: {e}")
            speak("I'm sorry, I didn't understand that. Please try again.")    

except KeyboardInterrupt:
    print("\nTranscription stopped by user.")
