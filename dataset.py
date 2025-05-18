from datasets import load_dataset
import time
from huggingface_hub import login
import os
import random
import soundfile as sf
import numpy as np

print("Loading Gigaspeech dataset...")

# Load the dataset with proper authentication
gs = load_dataset(
    "speechcolab/gigaspeech",
    "xl",
    trust_remote_code=True
)

print("\nDataset loaded successfully!")

# Create output directory if it doesn't exist
output_dir = "audio_files"
os.makedirs(output_dir, exist_ok=True)

# Get 10 random indices
num_samples = len(gs["train"])
random_indices = random.sample(range(num_samples), 10)

print(f"\nSaving 10 random audio samples to {output_dir}/")
for i, idx in enumerate(random_indices):
    # Get the audio sample
    audio_input = gs["train"][idx]["audio"]
    transcription = gs["train"][idx]["text"]
    
    # Save as WAV file
    output_path = os.path.join(output_dir, f"sample_{i+1}.wav")
    sf.write(output_path, audio_input["array"], audio_input["sampling_rate"])
    
    # Save transcription in a text file
    text_path = os.path.join(output_dir, f"sample_{i+1}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(transcription)
    
    print(f"Saved sample {i+1}: {output_path}")
    print(f"Transcription: {transcription[:100]}...")  # Print first 100 chars of transcription

print("\nDone! All files have been saved.")