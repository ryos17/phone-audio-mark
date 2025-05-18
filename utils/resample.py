import torch
import torchaudio
import argparse

def downsample_audio(input_path, output_path, target_sr=8000):
    # Load the audio file
    wav, sr = torchaudio.load(input_path)
    print(f"Original sample rate: {sr} Hz")
    
    # Create resampler
    resampler = torchaudio.transforms.Resample(sr, target_sr)
    
    # Resample the audio
    wav_resampled = resampler(wav)
    
    # Save the resampled audio
    torchaudio.save(output_path, wav_resampled, target_sr)
    print(f"Resampled audio saved to: {output_path}")
    print(f"New sample rate: {target_sr} Hz")
    print(f"Original shape: {wav.shape}")
    print(f"Resampled shape: {wav_resampled.shape}")

def main():
    parser = argparse.ArgumentParser(description='Resample audio to any sample rate')
    parser.add_argument('input_path', type=str, help='Path to input audio file')
    parser.add_argument('output_path', type=str, help='Path to save the resampled audio file')
    parser.add_argument('--target_sr', type=int, default=8000, help='Target sample rate (default: 8000)')
    
    args = parser.parse_args()
    
    downsample_audio(args.input_path, args.output_path, args.target_sr)

if __name__ == "__main__":
    main()
