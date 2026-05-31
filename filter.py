"""
filter.py
IIR filtering for the trigger track (bandpass) and target track (peaking EQ).
Core algorithms are based on the Audio EQ Cookbook by Robert Bristow-Johnson.
"""
import numpy as np
from typing import Tuple

def trigger_bandpass_coeffs(f_low: float, f_high: float, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2nd-order biquad bandpass coefficients via RBJ Audio EQ Cookbook.
    """
    fc = np.sqrt(f_low * f_high)
    q = fc / (f_high - f_low)

    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * q)
    
    # RBJ Bandpass formulation (constant peak gain variant)
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1 + alpha
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha
    
    # Normalize by a0 for discrete difference equation
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    
    return b, a

def target_eq_coeffs(fc: float, q: float, gain_db: float, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2nd-order biquad peaking EQ coefficients via RBJ Audio EQ Cookbook.
    """ 
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * q)
    
    # Half-power amplitude scaling for symmetric boost/cut
    A = 10 ** (gain_db / 40.0)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    
    # Normalize by a0 for discrete difference equation
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return b, a

def iir(
    block: np.ndarray, b: np.ndarray, a: np.ndarray, x_hist: np.ndarray, y_hist: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute IIR filtering block-by-block using Direct Form I.
    Return filtered block and updated history buffers.
    """
    output = np.zeros_like(block)
    order = len(b) - 1 
    
    for n in range(len(block)):
        xn = block[n]
        
        val_b = b[0] * xn
        for i in range(1, order + 1):
            val_b += b[i] * x_hist[i-1]
            
        val_a = 0.0
        for i in range(1, order + 1):
            val_a += a[i] * y_hist[i-1]
            
        yn = val_b - val_a
        output[n] = yn
        
        if order > 0:
            x_hist = np.roll(x_hist, 1)
            x_hist[0] = xn
            y_hist = np.roll(y_hist, 1)
            y_hist[0] = yn
            
    return output, x_hist, y_hist