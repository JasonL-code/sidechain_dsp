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

def frequency_center(audio_block: np.ndarray, sr: int, schmitt_thresh: float = 0.05) -> float:
    """
    Estimate fundamental frequency via Zero-Crossing Rate (ZCR) using a Schmitt Trigger.
    """
    crossings = 0
    if len(audio_block) == 0:
        return 0.0
        
    state = audio_block[0] > 0
    
    for sample in audio_block:
        if sample > schmitt_thresh and not state:
            state = True
            crossings += 1
        elif sample < -schmitt_thresh and state:
            state = False
            crossings += 1
            
    return (crossings / (2.0 * len(audio_block))) * sr