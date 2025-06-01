# Phone Quality Audio Watermarking

This repository contains code for audio watermarking experiments for CS224S final project (partnering with Sanas.ai)

## TODO
1. Training:
- figure out training with custom dataset (https://github.com/facebookresearch/audioseal/blob/main/docs/TRAINING.md)
- will have to use dora, and other tools, might be difficult formating the dataset to AudioCraft datasets
- once that is figured out, make a script to downsample to 8khz and train
- audiocraft dataset (https://github.com/facebookresearch/audiocraft/blob/main/docs/DATASETS.md)
- dora documentation (https://github.com/facebookresearch/dora)

2. Evaluation:
- figure out audiobenchmark evals
- simulate "attacks" and see case by case how decoding gets affected

## Setup

1. **Install Conda if you haven't already:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

2. **Create and activate the conda environment:**
```bash
conda env create -f environment.yml
conda activate audio-marking
```

### If you are training model...

3. **Install FFmpeg (required by AudioCraft):**
```bash
conda install "ffmpeg<5" -c conda-forge
```

4. **Clone AudioCraft repository (for AudioSeal training):**
```bash
git clone https://github.com/facebookresearch/audiocraft.git
cd audiocraft
pip install -e .
cd ..
```

## Project Structure

- `environment.yml`: Conda environment configuration
- `.gitignore`: Git ignore rules
- `README.md`: This file
- `utils/dataset.py`: Script to download and validate gigaspeech dataset
- `utils/encode.py`: Script to add watermark to audio files
- `utils/decode.py`: Script to detect watermark in audio files
- `utils/spectogram.py`: Script to generate and save spectrograms of audio files
- `utils/analyze.py`: Script to check audio file metadata (sample rate, channels, duration, format)

## Usage

### Analyzing Audio Files

To check the metadata of an audio file (sample rate, channels, duration, and format):
```bash
python utils/analyze.py path/to/audio.wav
```

The script will display:
- Sample rate in Hz
- Number of channels
- Duration in seconds
- Audio file format

### Adding Watermark to Audio

To add a watermark to an audio file:
```bash
python utils/encode.py --input_path path/to/audio.wav --sample_rate 16000
```

Optional: Add a 16-bit message to the watermark:
```bash
python utils/encode.py --input_path path/to/audio.wav --message "1010101010101010"
```

Optional: Specify custom output path:
```bash
python utils/encode.py --input_path path/to/audio.wav --output_path path/to/output.wav
```

### Detecting Watermark

To detect watermark in an audio file:
```bash
python utils/decode.py path/to/audio.wav
```

The script will output:
- Watermark probability (float number)
- Message (16-bit binary vector if watermarked)

### Generating Spectrograms

To generate spectrograms for one or more audio files:
```bash
python utils/spectogram.py path/to/audio1.wav path/to/audio2.wav
```

The script will:
- Create a `spectogram_files` directory if it doesn't exist
- Generate spectrograms for each input audio file
- Save spectrograms as PNG files with the same name as the input files
- Print information about each processed file

### Working with Gigaspeech Dataset

The `utils/dataset.py` script provides utilities for working with the Gigaspeech dataset. It will download and validate the dataset. Also gives 10 sample wav audio saved in `audio_files`:

```bash
python utils/dataset.py
```

## Training

### 1. Preparing GigaSpeech Dataset for Training
```bash
python prepare.py --size xs --output audiocraft/gigaspeech
```

Options:
- `--size`: Size of the GigaSpeech dataset ('xs', 's', 'm', 'l', 'xl')
- `--output`: Name of the output JSONL file (without extension)

The script will:
- Load the GigaSpeech dataset from HuggingFace
- Create a JSONL file in `audiocraft/gigaspeech/` directory
- Format each entry with required AudioCraft fields:
  - path: Path to audio file
  - duration: Audio duration in seconds
  - sample_rate: Sampling rate
  - amplitude: null
  - weight: null
  - info_path: null

### 2. Configure AudioCraft Training

Create the following datasource definition in `[audiocraft root]/configs/dset/audio/gigaspeech.yaml`:

```yaml
# @package __global__

datasource:
  max_sample_rate: 16000
  max_channels: 1

  train: gigaspeech
  valid: gigaspeech
  evaluate: gigaspeech
  generate: gigaspeech
```

### 3. Run Training
```bash
dora run solver=watermark/robustness dset=audio/gigaspeech
```

## Multi-GPU Training

To train using multiple GPUs, use the following command:

```bash
torchrun \
    --master-addr $(hostname -I | awk '{print $1}') \
    --master-port 29500 \
    --node_rank 0 \
    --nnodes 1 \
    --nproc-per-node 8 \  # Set to number of GPUs
    -m dora run \
    solver=watermark/robustness_8khz \
    dset=audio/gigaspeech_8khz_xl_half
```

Adjust `--nproc-per-node` to match your number of available GPUs.

## Evaluation

Create virtual environment
```bash
conda create -n audiomarkbench python=3.10 -y
conda activate audiomarkbench
```

Clone AudioMarkBench
```bash
git clone https://github.com/moyangkuo/AudioMarkBench/
```

Install requirements (skipping over uninstallable packages)
```bash
while IFS= read -r pkg; do                                               
  echo "Installing $pkg…"
  pip install "$pkg" || echo "  → Skipped $pkg"
done < requirements.txt
```

To install ViSQOL
```bash
brew install bazel
bazel build :visqol -c opt
```

Export this variable to suppress downsampling warnings
```bash
export GLOG_minloglevel=2
```

## Dependencies

See `environment.yml`




## Running Inference with Custom Trained Model

### 1. Prepare the Model

First, clone the AudioSeal repository and install required dependencies:

```bash
git clone https://github.com/facebookresearch/audioseal.git
pip install fire  # Required for checkpoint conversion
```

### 2. Convert Checkpoint

Convert your trained model checkpoint to the inference format:

```bash
python audioseal/src/scripts/checkpoints.py \
    --checkpoint=/path/to/checkpoint_50.th \
    --outdir=model_outputs \
    --suffix=model_name
```

### 3. Run Watermarking

Use the converted model to add a watermark to an audio file:

```bash
python utils/encode.py \
    --input_path path/to/input.wav \
    --message "1010101010101010" \
    --sample_rate 8000 \
    --output_path output.wav \
    --model_path model_outputs/checkpoint_generator_model_name.pth
```

### 4. Verify Watermark

Check if the watermark was successfully embedded:

```bash
python utils/decode.py output.wav
```