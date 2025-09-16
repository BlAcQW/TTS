# %% [markdown]
# Hausa TTS — Colab notebook (ready-to-run)
#
# What this notebook does (step-by-step):
# 1. mount Google Drive (optional) so you can save datasets and checkpoints
# 2. install dependencies (Coqui TTS, datasets, audio libs)
# 3. download or load a Hausa dataset (example: Mozilla Common Voice via HuggingFace)
# 4. preprocess audio and transcripts and create a LJSpeech-style metadata file
# 5. prepare a minimal Coqui TTS config and show how to start training (VITS/Glow/FastSpeech variants)
# 6. run inference on a trained checkpoint and play audio
#
# Notes:
# - This is a runnable Colab script. Open it in Colab, run cells top-to-bottom.
# - Training TTS end-to-end requires a GPU and time. Use Colab Pro/Local GPU if possible.
# - If you already have your Hausa dataset in Google Drive, point DATA_DIR to it and skip the Common Voice download cells.

# %% [markdown]
# 0) Quick setup: runtime
# - In Colab: Runtime > Change runtime type > GPU (preferably a Tesla T4/P100 or better)
# - Connect to runtime and then run cells below.

# %%
# 1) Mount Google Drive (optional but recommended)
from google.colab import drive
drive.mount('/content/drive')

# Set base paths - change these if you want to save elsewhere in Drive
BASE_DIR = '/content/hausa_tts'
DRIVE_SAVE_DIR = '/content/drive/MyDrive/hausa_tts'  # change to a folder in your Drive

import os
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
print('BASE_DIR:', BASE_DIR)
print('DRIVE_SAVE_DIR:', DRIVE_SAVE_DIR)

# %% [markdown]
# 2) Install system packages and Python deps
# - Coqui TTS (TTS) provides training and inference tools.
# - datasets for Common Voice loading; librosa/soundfile for audio ops.

# %%
# (run once)
!apt-get update -y -qq
!apt-get install -y -qq libsndfile1 ffmpeg

# Install python packages (this may take several minutes)
!pip install -q --upgrade pip
!pip install -q TTS datasets soundfile librosa numba pyopenjtalk num2words

# Verify `tts` CLI is available
!which tts || true

# %% [markdown]
# 3) (Option A) — Download Common Voice (Hausa) via HuggingFace `datasets`
# - If you already uploaded your Hausa dataset to Drive, skip this cell and set DATA_DIR to that location.
# - This example loads Common Voice (language code `ha`) and preps it into local wav files and metadata.

# %%
from datasets import load_dataset, Audio
import soundfile as sf
import librosa
import csv

# choose sampling rate for TTS (22050 is common)
TARGET_SR = 22050

print('Loading Common Voice (Hausa) dataset from HuggingFace...')
cv = load_dataset("mozilla-foundation/common_voice_11_0", "ha")
# available splits usually: train, validation, test
print('Splits:', list(cv.keys()))

# cast audio column to the target sampling rate so datasets will resample on-the-fly
for split in cv.keys():
    try:
        cv[split] = cv[split].cast_column("audio", Audio(sampling_rate=TARGET_SR))
    except Exception as e:
        print('cast_column skipped for', split, '->', e)

# %% [markdown]
# 4) Preprocess: save wav files and build `metadata.csv`
# - We save only reasonably short clips (0.5s < dur <= 15s) for TTS training convenience.
# - Metadata format used: `wav_path|transcript|speaker_id` (LJSpeech-like, accepted by Coqui examples)

# %%
import os
out_wav_dir = os.path.join(BASE_DIR, 'wavs')
os.makedirs(out_wav_dir, exist_ok=True)
meta_path = os.path.join(BASE_DIR, 'metadata.csv')

min_dur = 0.5
max_dur = 15.0

i = 0
with open(meta_path, 'w', encoding='utf-8') as mfile:
    for split in ['train', 'validation', 'test']:
        if split not in cv:
            continue
        ds = cv[split]
        print('Processing split', split, 'size', len(ds))
        for idx, item in enumerate(ds):
            # safety checks
            if item.get('sentence') is None:
                continue
            text = item['sentence'].strip()
            if len(text) < 1:
                continue
            audio = item['audio']
            if audio is None:
                continue
            arr = audio['array']
            sr = audio['sampling_rate']
            # ensure we have numpy
            try:
                import numpy as np
                arr = np.array(arr)
            except Exception:
                pass
            dur = len(arr) / TARGET_SR
            if dur < min_dur or dur > max_dur:
                continue
            # normalize text (basic) - lowercase and trim
            text_norm = text
            # write wav file (force PCM16)
            fname = f"cv_{split}_{i}.wav"
            out_path = os.path.join(out_wav_dir, fname)
            sf.write(out_path, arr, TARGET_SR, subtype='PCM_16')
            speaker = item.get('client_id', f"spk_{split}")
            mfile.write(f"{out_path}|{text_norm}|{speaker}\n")
            i += 1
print('Saved', i, 'wav files')
print('Metadata at', meta_path)

# Copy prepared data to Drive (optional)
!cp -r {out_wav_dir} {DRIVE_SAVE_DIR} || true
!cp {meta_path} {DRIVE_SAVE_DIR} || true

# %% [markdown]
# 5) Inspect dataset and check a sample

# %%
from IPython.display import Audio, display
# play first sample
with open(meta_path, 'r', encoding='utf-8') as f:
    first = f.readline().strip().split('|')
print('Sample metadata:', first)
display(Audio(first[0], rate=TARGET_SR))
print('Transcript:', first[1])

# %% [markdown]
# 6) Minimal Coqui TTS config (skeleton)
# - You should adapt this config to your model choice (vits, fastspeech2, glow-tts, etc.) and hardware.
# - See Coqui training docs and the Colab tutorial for full configs and explanations. Links are in the notebook header.

# %%
config_example = {
    "model": {
        "type": "vits",  # change to 'fastspeech2' or 'glowtts' if preferred
        "num_mels": 80,
        "audio": {"sample_rate": TARGET_SR}
    },
    "dataset": {
        "dataset_path": BASE_DIR,
        "metadata_filename": "metadata.csv",
        "audio": {"sample_rate": TARGET_SR}
    },
    "trainer": {
        "epochs": 200,
        "batch_size": 16,
        "learning_rate": 0.0001
    },
    "output_path": os.path.join(BASE_DIR, 'tts_output')
}

# Save a minimal config json (you will likely need to expand this for a real run)
import json
cfg_path = os.path.join(BASE_DIR, 'config_example.json')
with open(cfg_path, 'w', encoding='utf-8') as f:
    json.dump(config_example, f, indent=2)
print('Wrote example config to', cfg_path)

# %% [markdown]
# 7) Training — two options
# OPTION A (recommended for beginners): use pip-installed `TTS` trainer (simple)
#    - This uses the TTS package's training utilities. If the entrypoint is not available, use OPTION B.
# OPTION B (reproducible, matches docs): clone the Coqui TTS repo and run its `TTS/bin/train_tts.py` script.

# %% [markdown]
# 7A) (OPTION A) Try quick train command (may work after `pip install TTS`)
# - Note: you must edit `config_example.json` and set output paths / model settings for a full run.

# %%
# Example (uncomment to run):
# !python -m TTS.bin.train --config_path {cfg_path}

# %% [markdown]
# 7B) (OPTION B) Clone Coqui repo and run official training script
# - This is closest to the official tutorials and examples.

# %%
# (uncomment to run if you want to clone the repo)
# !git clone https://github.com/coqui-ai/TTS.git
# %cd TTS
# !pip install -r requirements.txt
# # run training (point --config_path to the config you created earlier)
# !python TTS/bin/train_tts.py --config_path {cfg_path}

# %% [markdown]
# 8) Inference (after you have a trained checkpoint)
# - Example CLI (Coqui provides `tts` command)
# - Replace `model.pth` and `config.json` with your trained checkpoint and config.

# %%
# Example (uncomment and edit paths):
# !tts --text "Sannu. Yaya aiki?" --model_path /content/hausa_tts/tts_output/best_model.pth --config_path /content/hausa_tts/config.json --out_path /content/hausa_tts/sample_out.wav
# from IPython.display import Audio
# Audio('/content/hausa_tts/sample_out.wav')

# %% [markdown]
# 9) Monitoring and logs
# - Coqui writes tensorboard logs; point TensorBoard at the output logs directory.
#
# Example:
# %load_ext tensorboard
# %tensorboard --logdir /content/hausa_tts/tts_output/logs

# %% [markdown]
# 10) Next steps & tips (short)
# - If you need single-speaker natural voice, use a single-speaker, studio-quality dataset (record 5-10 hours).
# - For multi-speaker datasets (like Common Voice), train a multi-speaker model and supply `speaker.json` mapping.
# - Text normalization for Hausa: numbers, abbreviations and punctuation should be expanded/normalized. Work with native speakers.
# - If data is scarce, consider fine-tuning a pre-trained multilingual TTS from Coqui or Hugging Face (faster convergence).

# %% [markdown]
# References
# - Coqui TTS documentation & training tutorial. See: https://colab.research.google.com/github/coqui-ai/TTS/blob/dev/notebooks/Tutorial_2_train_your_first_TTS_model.ipynb
# - Coqui TTS training docs: https://docs.coqui.ai/en/latest/training_a_model.html
# - HuggingFace Common Voice dataset: https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0
#
# Open the cells above and run them in Colab. Edit the config for your chosen model (VITS/FastSpeech2/etc.) and point DATA_DIR to your own Hausa dataset in Google Drive if you have it.
