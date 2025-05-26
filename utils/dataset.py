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

# Print dataset format and structure
print("\nDataset Format:")
print("=" * 50)
print("\nFeatures:")
print(gs["train"].features)
print("\nExample data structure:")
print("=" * 50)
example = gs["train"][0]
for key, value in example.items():
    if key == "audio":
        print("\naudio:")
        print("  - array shape:", value["array"].shape)
        print("  - array dtype:", value["array"].dtype)
        print("  - sampling_rate:", value["sampling_rate"])
        print("  - path:", value["path"])
        print("  - channels:", "stereo" if len(value["array"].shape) > 1 and value["array"].shape[1] == 2 else "mono")
    else:
        print(f"\n{key}:", value)

# Create output directory if it doesn't exist
output_dir = "audio_files"
os.makedirs(output_dir, exist_ok=True)

# Get 10 random indices from test set
num_samples = len(gs["test"])
random_indices = random.sample(range(num_samples), 10)

num_samples_train = len(gs["train"])
num_samples_val = len(gs["validation"])
num_samples_test = len(gs["test"])

print("\nDataset Statistics:")
print("=" * 50)
print(f"Number of training samples: {num_samples_train:,}")
print(f"Number of validation samples: {num_samples_val:,}")
print(f"Number of test samples: {num_samples_test:,}")

print(f"\nSaving 10 random audio samples to {output_dir}/")
for i, idx in enumerate(random_indices):
    # Get the audio sample
    audio_input = gs["test"][idx]["audio"]
    transcription = gs["test"][idx]["text"]
    
    # Save as WAV file
    output_path = os.path.join(output_dir, f"sample_{i+1}.wav")
    sf.write(output_path, audio_input["array"], audio_input["sampling_rate"])
    
    # Save transcription in a text file
    text_path = os.path.join(output_dir, f"sample_{i+1}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(transcription)
    
    print(f"Saved sample {i+1}: {output_path}")
    print(f"Transcription: {transcription}") 

print("\nDone! All files have been saved.")