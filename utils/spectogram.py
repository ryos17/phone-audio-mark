import argparse
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_spectrogram(wav, sr, title, ax):
    # Compute spectrogram
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=1024,
        hop_length=512,
        power=2.0
    )(wav)
    
    # Convert to log scale
    spectrogram = torch.log(spectrogram + 1e-9)
    
    # Convert to numpy for plotting
    spectrogram = spectrogram.squeeze().numpy()
    
    # Plot spectrogram
    im = ax.imshow(spectrogram, 
                  aspect='auto', 
                  origin='lower',
                  extent=[0, wav.shape[-1]/sr, 0, sr/2])
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    plt.colorbar(im, ax=ax, format='%+2.0f dB')

def process_audio_files(audio_paths):
    # Create output directory if it doesn't exist
    output_dir = "spectogram_files"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each audio file
    for audio_path in audio_paths:
        # Load audio
        wav, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Get filename for title and output
        filename = os.path.basename(audio_path)
        title = filename
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
        
        # Create figure for single spectrogram
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # Plot spectrogram
        plot_spectrogram(wav, sr, title, ax)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Processed: {audio_path}")
        print(f"Sample rate: {sr} Hz")
        print(f"Duration: {wav.shape[-1]/sr:.2f} seconds")
        print(f"Shape: {wav.shape}")
        print(f"Saved spectrogram to: {output_path}\n")

def main():
    parser = argparse.ArgumentParser(description='Plot spectrograms for audio files')
    parser.add_argument('audio_paths', nargs='+', help='Path(s) to audio file(s)')
    
    args = parser.parse_args()
    
    process_audio_files(args.audio_paths)

if __name__ == "__main__":
    main()