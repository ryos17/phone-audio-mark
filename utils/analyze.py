import soundfile as sf
import argparse
import os

def check_sample_rate(audio_path):
    """
    Check and print the sample rate of an audio file.
    
    Args:
        audio_path (str): Path to the audio file
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File '{audio_path}' does not exist.")
    
    # Read the audio file metadata
    info = sf.info(audio_path)
    
    # Print the information
    print(f"\nAudio file: {audio_path}")
    print(f"Sample rate: {info.samplerate} Hz")
    print(f"Channels: {info.channels}")
    print(f"Duration: {info.duration:.2f} seconds")
    print(f"Format: {info.format}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check audio file sample rate and metadata')
    parser.add_argument('audio_path', type=str, help='Path to the audio file')
    args = parser.parse_args()
    
    check_sample_rate(args.audio_path) 