"""
dynamics.py
Calculate RMS, compute gain reduction, and apply envelope smoothing for the compressor.
"""
import numpy as np

def rms_dbfs(audio_block: np.ndarray) -> float:
    """
    Calculate block RMS energy in dBFS.
    """
    EPSILON = 1e-10 # Prevent log10(0) error
    rms_val = np.sqrt(np.mean(audio_block ** 2) + EPSILON)
    
    return 20 * np.log10(rms_val)

def gain_reduction(current_db: float, previous_db: float, threshold_db: float, ratio: float) -> float:
    """
    Calculate target attenuation (<= 0.0 dB) using log-domain downward compression
    when the rate of change exceeds threshold.
    """
    delta_db = current_db - previous_db
    if delta_db > threshold_db:
        overshoot = delta_db - threshold_db
        return (overshoot / ratio) - overshoot
    return 0.0

def smooth(target_gain: float, prev_gain: float, attack_coeff: float, release_coeff: float) -> float:
    """
    Apply IIR smoothing to block gains.
    """
    if target_gain < prev_gain: 
        return attack_coeff * prev_gain + (1.0 - attack_coeff) * target_gain

    return release_coeff * prev_gain + (1.0 - release_coeff) * target_gain

def frequency_center(audio_block: np.ndarray, sr: int) -> float:
    """
    Estimate fundamental frequency via Zero-Crossing Rate (ZCR).
    """
    crossings = np.count_nonzero(np.diff(np.signbit(audio_block)))
    return (crossings / (2.0 * len(audio_block))) * sr