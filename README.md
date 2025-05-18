# Audio Marking Project

This repository contains code for audio watermarking experiments using AudioSeal and AudioMarkBench.

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

## Dependencies

The project uses the following main dependencies:
- AudioSeal
- AudioMarkBench
- PyTorch
- Other ML and audio processing libraries

See `environment.yml` for the complete list of dependencies.
