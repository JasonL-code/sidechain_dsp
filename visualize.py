"""
visualize.py
Logarithmic STFT visualization pipeline for sidechain compression verification.
"""
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def generate(
    original_bass: np.ndarray, 
    processed_bass: np.ndarray, 
    sr: int, 
    fmin: int = 20, 
    fmax: int = 1000, 
    hop_length: int = 512,
    save_path: str = "bass_spectrogram.png"
) -> None:
    """
    Generate a 3-panel logarithmic STFT visualization of original, processed, and attenuation delta signals.
    """
    # 4096 bins yield ~10.7 Hz spectral resolution, 
    # sufficient to show low-end frequency content
    n_fft = 4096

    stft_orig = np.abs(librosa.stft(original_bass, n_fft=n_fft, hop_length=hop_length))
    S_orig_db = librosa.amplitude_to_db(stft_orig, ref=np.max)
    
    stft_proc = np.abs(librosa.stft(processed_bass, n_fft=n_fft, hop_length=hop_length))
    # Anchor reference to original max to enforce a unified absolute dB scale across panels
    S_proc_db = librosa.amplitude_to_db(stft_proc, ref=np.max(stft_orig))
    
    S_diff = S_orig_db - S_proc_db

    # sharex=True locks temporal axes across panels for rigorous cross-domain frame inspection
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), layout="constrained", sharex=True)
    y_ticks = [20, 50, 100, 200, 500, 1000]
    
    # -30 dBFS floor isolates relevant high-energy low-end dynamics and rejects numerical noise floor
    db_min = -30.0
    
    img1 = librosa.display.specshow(
        S_orig_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log',
        ax=axes[0], cmap='magma', vmin=db_min, vmax=0
    )
    axes[0].set_title('Original Bass (Logarithmic Frequency Representation)', fontsize=12, pad=10)
    axes[0].set_ylim(fmin, fmax)
    axes[0].set_yticks(y_ticks)
    axes[0].get_yaxis().set_major_formatter(plt.ScalarFormatter())
    axes[0].set_ylabel('Frequency (Hz)')
    cbar1 = fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
    cbar1.ax.set_ylabel('Magnitude (dBFS)', rotation=270, labelpad=15)
    
    img2 = librosa.display.specshow(
        S_proc_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log',
        ax=axes[1], cmap='magma', vmin=db_min, vmax=0
    )
    axes[1].set_title('Processed Bass (Dynamic Gain Reduction Active)', fontsize=12, pad=10)
    axes[1].set_ylim(fmin, fmax)
    axes[1].set_yticks(y_ticks)
    axes[1].get_yaxis().set_major_formatter(plt.ScalarFormatter())
    axes[1].set_ylabel('Frequency (Hz)')
    cbar2 = fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
    cbar2.ax.set_ylabel('Magnitude (dBFS)', rotation=270, labelpad=15)
    
    # 12.0 dB display ceiling; corresponds to ~18 dB overshoot above threshold at ratio 3:1
    img3 = librosa.display.specshow(
        S_diff, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log',
        ax=axes[2], cmap='Reds', vmin=0, vmax=12.0
    )
    axes[2].set_title('Attenuation Footprint', fontsize=12, pad=10)
    axes[2].set_ylim(fmin, fmax)
    axes[2].set_yticks(y_ticks)
    axes[2].get_yaxis().set_major_formatter(plt.ScalarFormatter())
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_xlabel('Time (MM:SS)')
    cbar3 = fig.colorbar(img3, ax=axes[2], format='%+2.0f dB')
    cbar3.ax.set_ylabel('Reduction Magnitude (dB)', rotation=270, labelpad=15)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)