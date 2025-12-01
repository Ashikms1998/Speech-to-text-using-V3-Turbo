import os
import torch
import sounddevice as sd
import numpy as numpy
import scipy.io.wavfile as wav
from transformers import pipeline

# Define the folder where you pasted ffmpeg.exe
# We use r"" to make sure Windows backslashes don't cause errors
ffmpeg_folder = r"E:\learning-python\MLflow\speech-recognition-app"

# Add this folder to the Windows PATH variable
os.environ["PATH"] += os.pathsep + ffmpeg_folder
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading Whisper Turbo on {device} and torch_dtype {dtype}")

pipe = pipeline(
    "automatic-speech-recognition",
    model = "openai/whisper-large-v3-turbo",
    device = device,
    dtype  = dtype
)

#sample_audio_url = "https://www.ool.co.uk/wp-content/uploads/Spanish-A-Track-8.mp3"
#print("Listening to audio...")


# result = pipe(sample_audio_url,chunk_length_s = 30)

# print("\n--- Transcription ---\n")
# print(result["text"])
# print("\n---------------------\n")


#CONFIG FOR RECORDING

sample_rate = 16000 
duration = 5 #seconds
temp_file = "temp.wav"

def record_audio(duration):
    print("\n🔴Recording audio...")
    audio_data = sd.rec(int(duration*sample_rate), samplerate = sample_rate, channels=1 ,dtype="int16")
    sd.wait()
    print("Processing audio...")
    
    wav.write(temp_file,sample_rate,audio_data)

#THE INFINITE LOOP

print("Starting Live Transcription... (Press Ctrl+C to stop)")


try:
    while True:
        #Record audio
        record_audio(duration)

        #Transcribe audio generate_kwargs={"language": "english"} helps it decide faster if you only speak English
        result = pipe(temp_file)

        #Print Result
        text = result["text"]
        if text.strip() !="":
            print(f"You Said: {text}")

except KeyboardInterrupt:
    print("\nTranscription stopped by user.")