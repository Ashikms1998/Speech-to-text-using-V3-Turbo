import os

# Define the folder where you pasted ffmpeg.exe
# We use r"" to make sure Windows backslashes don't cause errors
ffmpeg_folder = r"E:\learning-python\MLflow\speech-recognition-app"

# Add this folder to the Windows PATH variable
os.environ["PATH"] += os.pathsep + ffmpeg_folder

import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading Whisper Turbo on {device} and torch_dtype {dtype}")

pipe = pipeline(
    "automatic-speech-recognition",
    model = "openai/whisper-large-v3-turbo",
    device = device,
    dtype  = dtype
)

sample_audio_url = "https://www.ool.co.uk/wp-content/uploads/Spanish-A-Track-8.mp3"

print("Listening to audio...")

result = pipe(sample_audio_url,chunk_length_s = 30)

print("\n--- Transcription ---\n")
print(result["text"])
print("\n---------------------\n")