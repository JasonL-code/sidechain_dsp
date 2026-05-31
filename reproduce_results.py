"""
reproduce_results.py
Automated replication script. Executes sidechain dynamic processing,
generates peak-normalized A/B testing mixdowns, and exports STFT delta plots.
"""
import os
import soundfile as sf
import numpy as np
import config
from main import process_stems
from visualize import generate

def mix_and_normalize(track_a: np.ndarray, track_b: np.ndarray, target_peak_db: float = -1.0) -> np.ndarray:
    """
    Sums two tracks and peak-normalizes to target_peak_db for valid A/B comparison.
    """ 
    mix = track_a + track_b
    peak_amplitude = np.max(np.abs(mix))
    
    if peak_amplitude > 0:
        target_linear = 10 ** (target_peak_db / 20.0)
        mix = mix * (target_linear / peak_amplitude)
        
    return mix

def run_reproduction(drum_wav: str, bass_wav: str, output_dir: str = "output") -> None:
    """
    Execute single-pass processing to generate processed stems, A/B mixes, and delta spectrograms.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("[Step 1] Executing DSP pipeline & exporting audio stems...")
    drum, bass, processed_bass = process_stems(drum_path=drum_wav, bass_path=bass_wav)
    
    processed_bass_path = os.path.join(output_dir, "bass_processed.wav")
    sf.write(processed_bass_path, processed_bass, config.SAMPLE_RATE)

    original_mix = mix_and_normalize(drum, bass, target_peak_db=-1.0)
    processed_mix = mix_and_normalize(drum, processed_bass, target_peak_db=-1.0)
    
    sf.write(os.path.join(output_dir, "ab_test_original_mix.wav"), original_mix, config.SAMPLE_RATE)
    sf.write(os.path.join(output_dir, "ab_test_processed_mix.wav"), processed_mix, config.SAMPLE_RATE)

    print("[Step 2] Generating STFT attenuation footprint visualization...")
    generate(
        original_bass=bass, 
        processed_bass=processed_bass, 
        sr=config.SAMPLE_RATE, 
        fmax=1000, 
        # Synchronize STFT frame rate with compressor control rate
        hop_length=config.BLOCK_SIZE, 
        save_path=os.path.join(output_dir, "bass_spectrogram.png")
    )
    print(f"Replication complete. Artifacts saved to: {output_dir}/")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    DRUM_INPUT = os.path.join(base_dir, "sample1", "Drum.wav")
    BASS_INPUT = os.path.join(base_dir, "sample1", "Bass.wav")
    OUTPUT_DIR = os.path.join(base_dir, "output")
    
    run_reproduction(DRUM_INPUT, BASS_INPUT, output_dir=OUTPUT_DIR)