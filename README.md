# Phone Quality Audio Watermarking

This repository contains code for audio watermarking experiments for CS224S final project. We will update the repository and README as needed so that our partner Sanas can easily use our scripts and see data, but all code for the class has been uploaded by the deadline.

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

Optional: Use a custom model (default: "audioseal_wm_16bits"):
```bash
python utils/encode.py --input_path path/to/audio.wav --model_path path/to/custom/generator_model.pth
```

### Detecting Watermark

To detect watermark in an audio file:
```bash
python utils/decode.py path/to/audio.wav
```

Optional: Specify a custom model path (default: "audioseal_detector_16bits"):
```bash
python utils/decode.py path/to/audio.wav --model_path path/to/custom/detector_model.pth
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

### Training Visualization Tools

#### 1. Single Run Analysis (`analyze_train.py`)

Visualize metrics from a single training run.

**Basic Usage:**
```bash
python utils/analyze_train.py <history_json> <metric_name> [options]
```

**Example: Plotting Discriminator Loss**
```bash
python utils/analyze_train.py dora/xps/a7a7d341/history.json d_loss \
    --output output \
    --name d_loss_a100 \
    --title "Loss for A₁₀₀" \
    --xlabel "Training Epochs" \
    --ylabel "Discriminator Loss" \
    --font-size 14
```

**Available Options:**
- `--output`: Output directory (default: 'outputs')
- `--name`: Output filename (without extension)
- `--title`: Plot title (supports Unicode subscripts, e.g., A₁₀₀)
- `--xlabel`: X-axis label (default: 'Epochs')
- `--ylabel`: Y-axis label (defaults to metric name)
- `--legend`: Legend labels (provide two space-separated values for train/val)
- `--font-size`: Base font size (default: 12)

#### 2. Multi-Run Comparison (`analyze_batch_train.py`)

Compare the same metric across multiple training runs in a single plot.

**Basic Usage:**
```bash
python utils/analyze_batch_train.py <history_json1> <history_json2> ... <metric_name> [options]
```

**Example: Comparing Discriminator Loss**
```bash
python utils/analyze_batch_train.py \
    dora/xps/6a28e352/history.json \
    dora/xps/a7a7d341/history.json \
    dora/xps/0427d672/history.json \
    d_loss \
    --output output \
    --name combined_d_loss \
    --title "Discriminator Loss" \
    --xlabel "Training Epochs" \
    --ylabel "Loss" \
    --legend "A₁₀" "A₁₀₀" "A₅₀₀₀" \
    --font-size 14 \
    --line-styles - - - \
    --epoch-limits 125 125 50
```

**Additional Options:**
- `--line-styles`: Line styles for each run (e.g., `- -- -` for solid, dashed, dash-dot)
- `--colors`: Custom colors for each run (hex codes)
- `--epoch-limits`: Maximum epochs to plot for each run (e.g., `125 125 50` for 125 epochs for first two runs, 50 for third)
- Other options same as `analyze_train.py`

**Notes:**
- Use Unicode subscripts (e.g., A₁₀, A₁₀₀) for clean formatting in titles and legends
- Default font is DeJavu Serif with Times New Roman fallback
- Output is saved as high-resolution PNG (300 DPI)

#### Notes

- The script automatically handles both training and validation metrics if available
- For subscripts in titles, use Unicode characters (e.g., A₁₀₀)
- Output is saved as a high-resolution PNG file (300 DPI)
- Use `--epoch-limits` to compare models trained for different numbers of epochs
- When using `--epoch-limits`, make sure the number of limits matches the number of input files

## Visqol Score Analysis

The `utils/visqol_stats.py` script calculates statistics for Visqol scores from audio mark evaluation results.

### Usage

```bash
python utils/visqol_stats.py <input_file>
```

### Example

```bash
python utils/visqol_stats.py eval_results/8khz_10hrs_125epochs.txt
```

### Output

The script will display the following statistics for the Visqol scores:
- Number of samples
- Maximum score
- Minimum score
- Mean score
- Median score
- Standard deviation

### Notes
- The script automatically detects the Visqol score column (case-insensitive)
- Values are truncated to 3 decimal places
- Handles various error cases (file not found, empty file, invalid format)

## Training

### 1. Configure AudioCraft Training Parameters

Replace `[audiocraft root]/configs` with our `config` folder. This contains the hyperparameters necessary for training in 8 kHz sampling rate.
```bash
# cd to phone-audio-mark root
cd ..

# copy config to audiocraft root
cp -r config/* audiocraft/config/
```

### 2. Preparing GigaSpeech Dataset for Training
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

### 3. Configure AudioCraft Dataset Format

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

### 4. Configure Dora Path
By default, checkpoints and inference files are saved in `/tmp/audiocraft_$USER/outputs`. However, to make our checkpoints more accessible, it is better to set custom path.

Create the following config definition in `[audiocraft root]/my_config.yaml`:

```yaml
# File name: my_config.yaml

default:
  dora_dir: /root/phone-audio-mark/dora
  partitions:
    global: your_slurm_partitions
    team: your_slurm_partitions
  reference_dir: /root/phone-audio-mark/dora/reference
```

### 5. Run Training
```bash
AUDIOCRAFT_CONFIG=my_config.yaml dora run solver=watermark/robustness dset=audio/gigaspeech
```

## Multi-GPU Training

To train using multiple GPUs, use the following command:
```bash
    torchrun  --master-addr $(hostname -I | awk '{print $1}')     --master-port 29500   --node_rank 0  --nnodes 1     --nproc-per-node 8  -m dora run    solver=watermark/robustness    dset=audio/gigaspeech_8khz_xl_half
```

Adjust `--nproc-per-node` to match your number of available GPUs. If you run into "opening too many files" errors, it is most likely the wandb artifacts so I recommend uninstalling wandb via `pip uninstall wandb`.

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

### VisQOL Setup

#### Install Bazelisk

```bash
# macOS (Homebrew)
brew install bazelisk

# Verify that it picks the version in .bazelversion
bazel version
```

#### Clone the ViSQOL repository
```bash
git clone https://github.com/google/visqol.git
cd visqol
```

#### Point Bazel at your Python interpreter
```bash
export PYTHON_BIN_PATH="$(which python)"
```

#### Clean any previous outputs and build the required packages
```bash
bazel clean --expunge
bazel build --action_env=PYTHON_BIN_PATH -c opt \
    //python:visqol_lib_py.so \
    //:similarity_result_py_pb2 \
    //:visqol_config_py_pb2
```

#### Install using pip
```bash
pip install -e .
```

#### To create the correct folder structure that AudioMarkBench expects:
```bash
cd <location_of_audiomarkbench>/AudioMarkBench/no-box

# 1) Copy the built Python package
cp -R ../../visqol/bazel-bin/python/visqol ./visqol

# 2) Ensure init files exist
touch visqol/__init__.py
touch visqol/pb2/__init__.py
```

#### Finally, your folder structure should look like this:
```md
.
├── AudioMarkBench/
└── └── no-box/
    ├── ├── nobox_audioseal_audiomarkdata.py
    └── └── visqol/                    ← copied package/
        ├── ├── __init__.py
        ├── ├── visqol_lib_py.so       ← native extension
        └── └── pb2/
            ├── ├── __init__.py
            ├── ├── similarity_result_pb2.py
            └── └── visqol_config_pb2.py
```

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
python utils/decode.py output.wav --model_path model_outputs/checkpoint_detector_model_name.pth
```

## References
- [AudioSeal Training Guide](https://github.com/facebookresearch/audioseal/blob/main/docs/TRAINING.md)
- [AudioCraft Datasets](https://github.com/facebookresearch/audiocraft/blob/main/docs/DATASETS.md)
- [Dora Documentation](https://github.com/facebookresearch/dora)
