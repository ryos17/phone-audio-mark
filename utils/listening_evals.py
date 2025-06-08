import os
import numpy as np

# Workaround for librosa compatibility with newer NumPy
np.complex = complex

import librosa
import matplotlib.pyplot as plt

# Try imports for PESQ and STOI, with fallback instructions
try:
    from pesq import pesq
except ModuleNotFoundError:
    raise ImportError("Please install PESQ: pip install pesq")

try:
    from pystoi import stoi
except ModuleNotFoundError:
    raise ImportError("Please install STOI: pip install pystoi")

# User-defined variables
path_model1 = "/Users/rushankgoyal/Desktop/Stanford/Sanas/AudioMarkBench/no-box/Evals/10hrs_125epochs_8khz_mp3/log_audioseal_audiomarkdata_max_5s/common_pert_mp3_bitrate_16/"
path_model2 = "/Users/rushankgoyal/Desktop/Stanford/Sanas/AudioMarkBench/no-box/Evals/5000hrs_50epochs_8khz_mp3/log_audioseal_audiomarkdata_max_5s/common_pert_mp3_bitrate_16/"
path_clean = "/Users/rushankgoyal/Desktop/Stanford/Sanas/AudioMarkBench/no-box/audiomarkdata/sample_20k/"  # Single folder with extra files
n_files = 2000   # Number of watermarked files per model to process
sampling_rate = 8000  # Narrowband rate for PESQ NB

# Metrics containers
metrics = {
    "model1": {"pesq": [], "stoi": []},
    "model2": {"pesq": [], "stoi": []}
}

def compute_metrics(wm_dir, clean_dir, n, sr):
    pesq_scores, stoi_scores = [], []
    wm_files = sorted([f for f in os.listdir(wm_dir) if f.lower().endswith(".wav")])[:n]
    i = 0
    
    for fname in wm_files:
        if i%100==0:
            print(i)
        i+=1
        # derive clean filename by removing last two underscore segments and changing to .mp3
        base = fname.rsplit('_', 2)[0]  # 'audiomarkdata_be_26791936'
        clean_fname = f"{base}.mp3"
        clean_path = os.path.join(clean_dir, clean_fname)
        wm_path = os.path.join(wm_dir, fname)
        
        if not os.path.exists(clean_path):
            continue  # skip if no matching clean file
        
        # load signals
        clean, _ = librosa.load(clean_path, sr=sr)
        wm, _    = librosa.load(wm_path, sr=sr)
        min_len = min(len(clean), len(wm))
        clean, wm = clean[:min_len], wm[:min_len]
        
        # compute PESQ NB
        try:
            p = pesq(sr, clean, wm, 'nb')
        except Exception:
            p = np.nan
        
        # compute STOI
        s = stoi(clean, wm, sr, extended=False)
        
        pesq_scores.append(p)
        stoi_scores.append(s)
    
    return np.array(pesq_scores), np.array(stoi_scores)

# Compute metrics for both models
metrics["model1"]["pesq"], metrics["model1"]["stoi"] = compute_metrics(path_model1, path_clean, n_files, sampling_rate)
metrics["model2"]["pesq"], metrics["model2"]["stoi"] = compute_metrics(path_model2, path_clean, n_files, sampling_rate)

# Print summary statistics
for model in ["model1", "model2"]:
    print(f"\n=== {model.upper()} ===")
    for name in ["pesq", "stoi"]:
        data = metrics[model][name]
        data = data[~np.isnan(data)]
        if len(data) == 0:
            print(f"{name.upper()}: no data")
            continue
        print(f"{name.upper()}: count={len(data)}, mean={np.mean(data):.3f}, std={np.std(data):.3f}, "
              f"min={np.min(data):.3f}, median={np.median(data):.3f}, max={np.max(data):.3f}")

# Plot histograms and boxplots
for name in ["pesq", "stoi"]:
    # Histogram Model 1
    plt.figure()
    plt.hist(metrics["model1"][name][~np.isnan(metrics["model1"][name])], bins=20)
    plt.title(f"{name.upper()} Histogram - Model 1")
    plt.xlabel(name.upper())
    plt.ylabel("Count")
    plt.show()
    
    # Histogram Model 2
    plt.figure()
    plt.hist(metrics["model2"][name][~np.isnan(metrics["model2"][name])], bins=20)
    plt.title(f"{name.upper()} Histogram - Model 2")
    plt.xlabel(name.upper())
    plt.ylabel("Count")
    plt.show()
    
    # Boxplot comparison
    plt.figure()
    plt.boxplot([
        metrics["model1"][name][~np.isnan(metrics["model1"][name])],
        metrics["model2"][name][~np.isnan(metrics["model2"][name])]
    ], labels=["Model 1", "Model 2"])
    plt.title(f"{name.upper()} Comparison Boxplot")
    plt.ylabel(name.upper())
    plt.show()
