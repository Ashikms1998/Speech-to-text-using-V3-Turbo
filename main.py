import os
from dotenv import load_dotenv
import sounddevice as sd
import numpy as numpy
import win32com.client
from groq import Groq
import soundfile as sf
import time

# Configure the GEMINI API key
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Define the folder where you pasted ffmpeg.exe
# We use r"" to make sure Windows backslashes don't cause errors
ffmpeg_folder = r"E:\learning-python\MLflow\speech-recognition-app"

# Add this folder to the Windows PATH variable
os.environ["PATH"] += os.pathsep + ffmpeg_folder
# Setup TTS Engine using Windows SAPI (more reliable than pyttsx3)
speaker = win32com.client.Dispatch("SAPI.SpVoice")
speaker.Rate = 1  # Speed: -10 to 10, default is 0


# CONFIG FOR RECORDING

sample_rate = 16000
threshold = 500
silence_limit = 1  # seconds
temp_file = "groq_temp.wav"


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
                audio_buffer.append(chunk)
                silence_start = None  # Reset silence start

            elif started_speaking:

                audio_buffer.append(chunk)

                if silence_start is None:
                    silence_start = time.time()

                # Check if we have been silent for too long

                if time.time() - silence_start > silence_limit:
                    started_speaking = False
                    break  # Stop recording

            else:
                pass  # We haven't started speaking yet, just ignore the noise

    # Save the audio to a temporary file
    audio_numpy = numpy.concatenate(audio_buffer)
    sf.write(temp_file, audio_numpy, sample_rate)
    return True


# MAIN LOOP

speak("Groq Listening Mode Online.")

try:
    while True:
        # This function now BLOCKS until you finish a sentence
        smart_record_audio()

        # STEP A: Groq Whisper (The Ear)
        # We send the file to Groq's cloud. It comes back almost instantly

        with open(temp_file, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_file, file.read()),
                model="whisper-large-v3",
            )

        user_text = transcription.text

        if user_text.strip() != "":
            print(f"You: {user_text}")

            # The Logic for AI model
            # This is where you would normally send 'user_text' to Gemini/OpenAI.
            # For now, we use simple Python logic:

            if "stop" in user_text.lower() or "exit" in user_text.lower():
                speak("Goodbye! Have a great day!")
                break

            # Groq LLM (The Brain)
            # Using Llama3-8b because it is super fast on Groq

            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Reply in one short sentence."},
                    {"role": "user", "content": user_text},
                ],
            )
            ai_reply = completion.choices[0].message.content
            print(f"🤖 AI: {ai_reply}")
            speak(ai_reply)

except KeyboardInterrupt:
    print("\nTranscription stopped by user.")
