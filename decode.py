import argparse
import torch
import torchaudio
from audioseal import AudioSeal

def decode_audio(input_path):
    # Load the audio file
    wav, sr = torchaudio.load(input_path)
    
    # Ensure correct shape (batch, channels, samples)
    if wav.dim() == 2:  # If stereo, convert to mono
        wav = wav.mean(dim=0, keepdim=True)
    if wav.dim() == 1:  # If mono without channel dimension
        wav = wav.unsqueeze(0)
    wav = wav.unsqueeze(0)  # Add batch dimension
    
    print(f"Audio shape: {wav.shape}")  # Should be (1, 1, samples)
    
    # Load the detector model
    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    
    # Detect watermark
    result, message = detector.detect_watermark(wav, sr)
    
    # Print results
    print(f"\nResults for: {input_path}")
    print(f"Watermark probability: {result}")  # Result is a float number
    print(f"Message: {message}")  # Message is a binary vector of 16 bits

def main():
    parser = argparse.ArgumentParser(description='Detect watermark in audio file')
    parser.add_argument('input_path', type=str, help='Path to input audio file')
    
    args = parser.parse_args()
    
    decode_audio(args.input_path)

if __name__ == "__main__":
    main() 