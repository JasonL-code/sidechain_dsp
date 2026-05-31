"""
audio_io.py
Load and prepare audio tracks for processing.
"""
import os
from typing import Tuple, Optional
import numpy as np
import librosa

def load_and_align(
    drum_path: Optional[str] = None, 
    bass_path: Optional[str] = None, 
    vocal_path: Optional[str] = None, 
    other_path: Optional[str] = None, 
    target_sr: int = 44100
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """ 
    Load tracks as mono, resample, truncate and align track lengths, 
    return tracks as tuple of arrays.
    """
    paths = {
        "drum": drum_path,
        "bass": bass_path,
        "vocal": vocal_path,
        "other": other_path
    }
    
    tracks_raw = {}
    
    for name, path in paths.items():
        if path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"[I/O Error] Missing {name} track: {path}")

            y, _ = librosa.load(path, sr=target_sr, mono=True)
            tracks_raw[name] = y

    if not tracks_raw:
        raise ValueError("[I/O Error] No active tracks provided.")

    min_len = min(len(y) for y in tracks_raw.values())

    return (
        tracks_raw["drum"][:min_len] if "drum" in tracks_raw else None,
        tracks_raw["bass"][:min_len] if "bass" in tracks_raw else None,
        tracks_raw["vocal"][:min_len] if "vocal" in tracks_raw else None,
        tracks_raw["other"][:min_len] if "other" in tracks_raw else None
    )