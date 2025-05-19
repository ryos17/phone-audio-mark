import argparse
import torch
import torchaudio
from audioseal import AudioSeal
import os

def encode_audio(input_path, sample_rate, message=None, output_path=None):
    # Load the audio file
    wav, sr = torchaudio.load(input_path)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        print(f"Audio is now resampled to {sample_rate} Hz from {sr} Hz")
        wav = resampler(wav)
    
    # Ensure correct shape (batch, channels, samples)
    if wav.dim() == 2:  # If stereo, convert to mono
        wav = wav.mean(dim=0, keepdim=True)
    if wav.dim() == 1:  # If mono without channel dimension
        wav = wav.unsqueeze(0)
    wav = wav.unsqueeze(0)  # Add batch dimension
    
    print(f"Audio shape: {wav.shape}")  # Should be (1, 1, samples)
    
    # Load the watermarking model
    model = AudioSeal.load_generator("audioseal_wm_16bits")
    
    # Generate watermark with optional message
    if message is not None:
        # Convert message to binary tensor with correct shape (batch_size, nbits)
        msg_bits = torch.tensor([int(bit) for bit in message], dtype=torch.float32)
        msg_bits = msg_bits.unsqueeze(0)  # Add batch dimension to match (batch_size, nbits)
        watermark = model.get_watermark(wav, sample_rate, message=msg_bits)
        print(f"Embedding message: {message}")
    else:
        watermark = model.get_watermark(wav, sample_rate)
        print("No message embedded in watermark")
    
    # Add watermark to original audio
    watermarked_audio = wav + watermark
    
    # Create output filename if not provided
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_encoded.wav"
    
    # Save the watermarked audio (remove batch dimension and detach for saving)
    torchaudio.save(output_path, watermarked_audio.squeeze(0).detach(), sample_rate)
    print(f"Watermarked audio saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Add watermark to audio file')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input audio file')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate for model (default: 16000)')
    parser.add_argument('--message', type=str, help='16-bit binary message to embed (e.g., "1010101010101010")')
    parser.add_argument('--output_path', type=str, help='Path to save the watermarked audio file (default: input_path_encoded.wav)')
    
    args = parser.parse_args()
    
    # Validate message if provided
    if args.message:
        if len(args.message) != 16 or not all(bit in '01' for bit in args.message):
            print("Error: Message must be exactly 16 bits (0s and 1s)")
            return
    
    encode_audio(args.input_path, args.sample_rate, args.message, args.output_path)

if __name__ == "__main__":
    main() 