import argparse
import torch
import torchaudio
from audioseal import AudioSeal

def decode_audio(input_path):
    # Load the audio file
    wav, sr = torchaudio.load(input_path)
    
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