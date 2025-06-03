# vad.py
import torch
import numpy as np

# Load Silero VAD model and utilities from torch hub
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, _, read_audio, _, _) = utils

def is_speech(audio_bytes: bytes, sample_rate=16000) -> bool:
    audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
    audio_tensor = torch.tensor(audio_np, dtype=torch.float32)
    timestamps = get_speech_timestamps(audio_tensor, model, sampling_rate=sample_rate)
    return len(timestamps) > 0
