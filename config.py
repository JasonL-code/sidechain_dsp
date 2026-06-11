"""
config.py
Global constants and DSP presets.
"""

SAMPLE_RATE = 44100
BLOCK_SIZE = 512

# Control rate is used to discretize continuous-time constants (ms) for block-level smoothing.
CONTROL_RATE = SAMPLE_RATE / BLOCK_SIZE

# Sidechain Pair Preset Format: PRESET_[TRIGGER]_[TARGET]

PRESET_DRUM_BASS = {
    "trigger_bp_low": 60.0, # Hz
    "trigger_bp_high": 120.0, # Hz
    "threshold": 6.0, # dBFS
    
    "fc_min": 40.0, # Hz
    "fc_max": 200.0, # Hz
    "target_q": 1.0, # Q-factor (dimensionless)
    
    "ratio": 3.0, # :1 ratio
    "attack_ms": 5.0, # ms
    "release_ms": 40.0, # ms
    
    "schmitt_thresh": 0.05,
}