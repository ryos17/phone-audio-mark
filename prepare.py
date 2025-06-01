from datasets import load_dataset
import os
import json
from tqdm import tqdm
import argparse
import soundfile as sf
import librosa
import multiprocessing as mp
from functools import partial
import numpy as np
import gc
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available - memory monitoring disabled")

def get_memory_usage():
    """Get current memory usage in MB"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    return 0

def process_audio_item(item, audio_output_dir, target_sr=8000):
    """Process a single audio item - for multiprocessing"""
    try:
        # Get original audio data
        audio_array = item["audio"]["array"]
        original_sr = item["audio"]["sampling_rate"]
        duration = item["end_time"] - item["begin_time"]
        
        # Determine if audio is stereo or mono
        is_stereo = len(audio_array.shape) > 1 and audio_array.shape[1] == 2
        channels = 2 if is_stereo else 1
        
        # Downsample audio to target sample rate (much faster with librosa)
        if original_sr != target_sr:
            # Use faster resampling method
            audio_resampled = librosa.resample(
                audio_array.astype(np.float32), 
                orig_sr=original_sr, 
                target_sr=target_sr,
                res_type='kaiser_fast'  # Faster resampling
            )
        else:
            audio_resampled = audio_array.astype(np.float32)
        
        # Create filename for the downsampled audio
        original_filename = os.path.basename(item["audio"]["path"])
        filename_without_ext = os.path.splitext(original_filename)[0]
        new_filename = f"{filename_without_ext}_8khz.wav"
        
        # Determine split from the item (you might need to adjust this logic)
        split = item.get('split', 'train')  # Default to train if no split info
        split_dir = os.path.join(audio_output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        local_audio_path = os.path.join(split_dir, new_filename)
        
        # Save downsampled audio
        sf.write(local_audio_path, audio_resampled, target_sr)
        
        # Create absolute path for JSONL
        absolute_audio_path = os.path.abspath(local_audio_path)
        
        # Return entry data
        return {
            "path": absolute_audio_path,
            "duration": duration,
            "sample_rate": target_sr,
            "channels": channels,
            "amplitude": None,
            "weight": None,
            "info_path": None,
            "is_stereo": is_stereo
        }
    except Exception as e:
        print(f"Error processing {item.get('audio', {}).get('path', 'unknown')}: {e}")
        return None

def safe_get_item(dataset, split, index):
    """Safely get an item from dataset, handling corrupted files"""
    try:
        return dataset[split][index]
    except Exception as e:
        print(f"Skipping corrupted file at index {index} in {split}: {e}")
        return None

def prepare_gigaspeech_jsonl(dataset_size, output_dir, audio_dir=None, max_hours=None, num_workers=None):
    """
    Prepare GigaSpeech dataset for AudioCraft by creating a JSONL file
    with the required format and downsampling audio to 8kHz:
    {
        "path": path to downsampled audio file,
        "duration": audio duration in seconds,
        "sample_rate": 8000,
        "channels": number of channels (1 for mono, 2 for stereo),
        "amplitude": null,
        "weight": null,
        "info_path": null
    }
    Args:
        dataset_size: Size of the dataset ('xs', 's', 'm', 'l', 'xl')
        output_dir: Directory path where to save the JSONL file
        audio_dir: Directory path where to save the audio files (optional)
        max_hours: Maximum hours of audio to process (optional)
        num_workers: Number of parallel workers (optional, defaults to CPU count)
    """
    print(f"Loading GigaSpeech dataset (size: {dataset_size})...")
    
    # Load the dataset
    gs = load_dataset(
        "speechcolab/gigaspeech",
        dataset_size,
        trust_remote_code=True
    )
    
    print("Dataset loaded successfully!")
    
    # Create output directories
    jsonl_dir = output_dir
    
    # Use custom audio directory or default
    if audio_dir:
        audio_output_dir = audio_dir
    else:
        audio_output_dir = f"gigaspeech_data/{dataset_size}"
        if max_hours:
            audio_output_dir += f"_{max_hours}h"
    
    os.makedirs(jsonl_dir, exist_ok=True)
    os.makedirs(audio_output_dir, exist_ok=True)
    
    # Path for the JSONL file
    jsonl_path = os.path.join(jsonl_dir, "data.jsonl")
    print(f"\nCreating JSONL file at: {jsonl_path}")
    print(f"Saving downsampled audio to: {audio_output_dir}")
    if max_hours:
        print(f"Maximum hours to process: {max_hours}")
    
    # Check if we're resuming from a previous run
    resume_from = 0
    existing_duration = 0.0
    if os.path.exists(jsonl_path):
        print("Found existing JSONL file - checking for resume...")
        try:
            with open(jsonl_path, 'r') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        entry = json.loads(line.strip())
                        existing_duration += entry.get('duration', 0)
                        resume_from = line_num + 1
            print(f"Resuming from entry {resume_from} ({existing_duration/3600:.2f} hours already processed)")
        except Exception as e:
            print(f"Error reading existing file: {e}")
            print("Starting fresh...")
            resume_from = 0
            existing_duration = 0.0
    
    # Set number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 4)  # Reduce to 4 to save memory
    print(f"Using {num_workers} parallel workers")
    
    # Track statistics
    total_duration = existing_duration  # Start from existing duration if resuming
    max_duration = max_hours * 3600 if max_hours else float('inf')
    skipped_files = 0
    mono_count = 0
    stereo_count = 0
    processed_count = resume_from  # Start count from resume point
    items_to_skip = resume_from  # Track how many items to skip
    
    # Open JSONL file for writing (append if resuming)
    file_mode = 'a' if resume_from > 0 else 'w'
    with open(jsonl_path, file_mode) as jsonl_file:
        print("Processing items in streaming mode...")
        
        # Process each split
        for split in ['train', 'validation', 'test']:
            split_dir = os.path.join(audio_output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            
            print(f"Processing {split} split...")
            split_length = len(gs[split])
            
            # Process in smaller batches to manage memory
            batch_size = 50  # Smaller batch size for memory efficiency
            
            for start_idx in tqdm(range(0, split_length, batch_size), desc=f"Processing {split}"):
                if total_duration >= max_duration:
                    break
                    
                end_idx = min(start_idx + batch_size, split_length)
                batch_items = []
                
                # Collect batch items
                for idx in range(start_idx, end_idx):
                    if total_duration >= max_duration:
                        break
                        
                    try:
                        item = safe_get_item(gs, split, idx)
                        if item is None:
                            skipped_files += 1
                            continue
                            
                        duration = item["end_time"] - item["begin_time"]
                        
                        # Skip items if we're resuming
                        if items_to_skip > 0:
                            items_to_skip -= 1
                            continue
                        
                        if total_duration + duration > max_duration:
                            break
                            
                        # Add split info to item for processing
                        item_with_split = dict(item)
                        item_with_split['split'] = split
                        batch_items.append(item_with_split)
                        total_duration += duration
                        
                    except Exception as e:
                        print(f"Error accessing item {idx} in {split}: {e}")
                        skipped_files += 1
                        continue
                
                if not batch_items:
                    continue
                
                # Process batch in parallel
                process_func = partial(process_audio_item, audio_output_dir=audio_output_dir, target_sr=8000)
                
                if num_workers > 1:
                    with mp.Pool(num_workers) as pool:
                        results = pool.map(process_func, batch_items)
                else:
                    # Single-threaded processing for debugging
                    results = [process_func(item) for item in batch_items]
                
                # Write results immediately to JSONL
                for result in results:
                    if result is not None:
                        # Remove the is_stereo field before writing
                        entry_clean = {k: v for k, v in result.items() if k != "is_stereo"}
                        jsonl_file.write(json.dumps(entry_clean) + '\n')
                        jsonl_file.flush()  # Ensure data is written immediately
                        
                        processed_count += 1
                        if result["is_stereo"]:
                            stereo_count += 1
                        else:
                            mono_count += 1
                
                # Clear batch from memory
                del batch_items
                del results
                
                # Print progress every 1000 items
                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count:,} items ({total_duration/3600:.2f} hours)")
                
                # Memory management - force garbage collection every 500 items
                if processed_count % 500 == 0:
                    gc.collect()
                    if PSUTIL_AVAILABLE:
                        memory_mb = get_memory_usage()
                        print(f"Memory usage: {memory_mb:.1f} MB")
            
            if total_duration >= max_duration:
                break
    
    print("\nJSONL file created successfully!")
    print("\nAudio Format Statistics:")
    print(f"Mono files: {mono_count:,}")
    print(f"Stereo files: {stereo_count:,}")
    print(f"Total files: {processed_count:,}")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"All audio downsampled to: 8000 Hz")
    print(f"Audio files saved in: {os.path.abspath(audio_output_dir)}")
    if skipped_files > 0:
        print(f"Skipped files due to errors: {skipped_files}")
    
    # Print sample entries by reading back from the file
    print("\nSample entries from JSONL file:")
    print("=" * 50)
    try:
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Only show first 3 entries
                    break
                entry = json.loads(line.strip())
                print(f"\nEntry {i+1}:")
                print(entry)
    except Exception as e:
        print(f"Could not read sample entries: {e}")

def main():
    parser = argparse.ArgumentParser(description='Prepare GigaSpeech dataset for AudioCraft with 8kHz downsampling')
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
        default='audiocraft/gigaspeech',
        help='Directory path where to save the JSONL file (e.g., audiocraft/gigaspeech_8khz_100)'
    )
    parser.add_argument(
        '--audio_dir',
        type=str,
        help='Directory path where to save the audio files (e.g., gigaspeech_data/hour_100)'
    )
    parser.add_argument(
        '--max_hours',
        type=float,
        help='Maximum hours of audio to process (optional)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        help='Number of parallel workers (default: CPU count, max 8)'
    )
    
    args = parser.parse_args()
    prepare_gigaspeech_jsonl(args.size, args.output, args.audio_dir, args.max_hours, args.workers)

if __name__ == "__main__":
    main() 