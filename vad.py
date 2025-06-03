# vad.py
import numpy as np
import torch
from silero_vad import VoiceActivityDetector

# Load Silero VAD model
vad = VoiceActivityDetector("cpu")  # Or "cuda" if on GPU

def is_speech(audio_bytes: bytes, sample_rate=16000) -> bool:
    audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
    speech_probs = vad(audio_np, sample_rate)
    return any(prob > 0.5 for prob in speech_probs)
