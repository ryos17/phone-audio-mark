from datasets import load_dataset
import os
import json
from tqdm import tqdm
import argparse

def prepare_gigaspeech_jsonl(dataset_size, output_name):
    """
    Prepare GigaSpeech dataset for AudioCraft by creating a JSONL file
    with the required format:
    {
        "path": path to audio file,
        "duration": audio duration in seconds,
        "sample_rate": sampling rate,
        "channels": number of channels (1 for mono, 2 for stereo),
        "amplitude": null,
        "weight": null,
        "info_path": null
    }
    Args:
        dataset_size: Size of the dataset ('xs', 's', 'm', 'l', 'xl')
        output_name: Name of the output JSONL file (without extension)
    """
    print(f"Loading GigaSpeech dataset (size: {dataset_size})...")
    
    # Load the dataset
    gs = load_dataset(
        "speechcolab/gigaspeech",
        dataset_size,
        trust_remote_code=True
    )
    
    print("Dataset loaded successfully!")
    
    # Create output directory
    jsonl_dir = "audiocraft/gigaspeech"
    os.makedirs(jsonl_dir, exist_ok=True)
    
    # Path for the JSONL file
    jsonl_path = os.path.join(jsonl_dir, f"{output_name}.jsonl")
    print(f"\nCreating JSONL file at: {jsonl_path}")
    
    # Get a sample audio path to determine the root directory
    sample_path = None
    for split in ['train', 'validation', 'test']:
        if len(gs[split]) > 0:
            sample_path = gs[split][0]['audio']['path']
            break
    
    # Extract the root directory from the sample path
    root_dir = os.path.dirname(sample_path) if sample_path else None
    
    # Count total items for progress bar
    total_items = sum(len(gs[split]) for split in ['train', 'validation', 'test'])
    
    # Track audio format statistics
    mono_count = 0
    stereo_count = 0
    
    with open(jsonl_path, 'w') as f:
        # Use tqdm for progress bar
        with tqdm(total=total_items, desc="Processing entries") as pbar:
            # Process each split
            for split in ['train', 'validation', 'test']:
                for item in gs[split]:
                    # Determine if audio is stereo or mono
                    audio_array = item["audio"]["array"]
                    is_stereo = len(audio_array.shape) > 1 and audio_array.shape[1] == 2
                    channels = 2 if is_stereo else 1
                    
                    if is_stereo:
                        stereo_count += 1
                    else:
                        mono_count += 1
                    
                    # Create entry in required format
                    entry = {
                        "path": item["audio"]["path"],
                        "duration": (item["end_time"] - item["begin_time"]),
                        "sample_rate": item["audio"]["sampling_rate"],
                        "channels": channels,
                        "amplitude": None,
                        "weight": None,
                        "info_path": None
                    }
                    # Write to JSONL file
                    f.write(json.dumps(entry) + '\n')
                    pbar.update(1)
    
    print("\nJSONL file created successfully!")
    print("\nAudio Format Statistics:")
    print(f"Mono files: {mono_count:,}")
    print(f"Stereo files: {stereo_count:,}")
    print(f"Total files: {mono_count + stereo_count:,}")
    
    # Print sample entries
    print("\nSample entries from JSONL file:")
    print("=" * 50)
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 3:  # Print first 3 entries
                print(f"\nEntry {i+1}:")
                print(json.loads(line))
            else:
                break

def main():
    parser = argparse.ArgumentParser(description='Prepare GigaSpeech dataset for AudioCraft')
    parser.add_argument(
        '--size',
        type=str,
        choices=['xs', 's', 'm', 'l', 'xl'],
        default='xs',
        help='Size of the GigaSpeech dataset (xs, s, m, l, xl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data',
        help='Name of the output JSONL file (without extension)'
    )
    
    args = parser.parse_args()
    prepare_gigaspeech_jsonl(args.size, args.output)

if __name__ == "__main__":
    main() 