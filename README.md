# Audio Marking Project

This repository contains code for audio watermarking experiments for CS224S final project (partnering with Sanas.ai)

## Setup

1. Install Conda if you haven't already:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate audio-marking
```

## Project Structure

- `environment.yml`: Conda environment configuration
- `.gitignore`: Git ignore rules
- `README.md`: This file
- `encode.py`: Script to add watermark to audio files
- `decode.py`: Script to detect watermark in audio files
- `utils/spectogram.py`: Script to generate and save spectrograms of audio files

## Usage

### Adding Watermark to Audio

To add a watermark to an audio file:
```bash
python encode.py --input_path path/to/audio.wav --sample_rate 16000
```

Optional: Add a 16-bit message to the watermark:
```bash
python encode.py --input_path path/to/audio.wav --message "1010101010101010"
```

Optional: Specify custom output path:
```bash
python encode.py --input_path path/to/audio.wav --output_path path/to/output.wav
```

### Detecting Watermark

To detect watermark in an audio file:
```bash
python decode.py path/to/audio.wav
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

## Dependencies

The project uses the following main dependencies:
- AudioSeal
- AudioMarkBench
- PyTorch
- torchaudio
- matplotlib
- numpy
- Other ML and audio processing libraries

See `environment.yml` for the complete list of dependencies.
