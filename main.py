"""
main.py
Block-based offline DSP pipeline for sidechain dynamic range compression.
Simulates a streaming environment by processing contiguous audio frames sequentially.
"""
from typing import Tuple
import numpy as np

import audio_io
import config
import dynamics
import filter

def process_stems(
    drum_path: str, bass_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Block-by-block sidechain compression via time-domain biquad filtering.
    Propagates filter history across boundaries to ensure phase continuity.
    """

    drum, bass, _, _ = audio_io.load_and_align(
        drum_path=drum_path, 
        bass_path=bass_path, 
        target_sr=config.SAMPLE_RATE
    )

    total_samples = len(drum)
    block_size = config.BLOCK_SIZE
    output_bass = np.zeros_like(bass)

    bp_b, bp_a = filter.trigger_bandpass_coeffs(
        config.PRESET_DRUM_BASS["trigger_bp_low"], 
        config.PRESET_DRUM_BASS["trigger_bp_high"], 
        config.SAMPLE_RATE
    )
    
    bp_order = len(bp_b) - 1
    # Initialize with first sample to suppress DC transient at filter startup
    drum_x_hist = np.full(bp_order, drum[0])
    drum_y_hist = np.full(bp_order, drum[0])
    
    bass_x_hist = None
    bass_y_hist = None 
    
    smoothed_gain = 0.0
    # Initialize log-domain energy state to prevent cold-start false trigger
    prev_energy_db = None
    
    attack_coeff = np.exp(-1.0 / (config.CONTROL_RATE * config.PRESET_DRUM_BASS["attack_ms"] / 1000.0))
    release_coeff = np.exp(-1.0 / (config.CONTROL_RATE * config.PRESET_DRUM_BASS["release_ms"] / 1000.0))

    for i in range(0, total_samples, block_size):
        drum_block = drum[i:i+block_size]
        bass_block = bass[i:i+block_size]
        
        if len(drum_block) < block_size:
            break
            
        # --- Detector Path (Control-Rate Extraction) ---
        filtered_drum, drum_x_hist, drum_y_hist = filter.iir(
            drum_block, bp_b, bp_a, drum_x_hist, drum_y_hist
        )
        
        current_db = dynamics.rms_dbfs(filtered_drum)
        
        if prev_energy_db is None:
            prev_energy_db = current_db
            
        fc = dynamics.frequency_center(
            filtered_drum, 
            config.SAMPLE_RATE,
            config.PRESET_DRUM_BASS["schmitt_thresh"]
        )
        dynamic_fc = np.clip(
            fc, 
            config.PRESET_DRUM_BASS["fc_min"], 
            config.PRESET_DRUM_BASS["fc_max"]
        )
        
        target_gain = dynamics.gain_reduction(
            current_db, 
            prev_energy_db,
            config.PRESET_DRUM_BASS["threshold"], 
            config.PRESET_DRUM_BASS["ratio"]
        )
          
        # --- Smoothing Stage ---
        smoothed_gain = dynamics.smooth(
            target_gain, smoothed_gain, attack_coeff, release_coeff
        )
        
        # --- Processor Path  ---
        eq_b, eq_a = filter.target_eq_coeffs(
            dynamic_fc,
            config.PRESET_DRUM_BASS["target_q"],
            smoothed_gain,
            config.SAMPLE_RATE
        )
        
        if bass_x_hist is None:
            eq_order = len(eq_b) - 1
            bass_x_hist = np.full(eq_order, bass_block[0])
            bass_y_hist = np.full(eq_order, bass_block[0])
            
        processed_bass_block, bass_x_hist, bass_y_hist = filter.iir(
            bass_block, eq_b, eq_a, bass_x_hist, bass_y_hist
        )
        
        output_bass[i:i+block_size] = processed_bass_block
        
        prev_energy_db = current_db
        
        if smoothed_gain < -1.0:
            timestamp = i / config.SAMPLE_RATE
            print(f"[Time: {timestamp:.2f}s] Gain: {smoothed_gain:.1f} dB | FC: {dynamic_fc:.1f} Hz")
            
    return drum, bass, output_bass