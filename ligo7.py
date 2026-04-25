import numpy as np

def analyze_tiling_ratios(peaks, f_target):
    """
    Checks if the observed 'Jitter' peaks follow aperiodic scaling 
    relative to the Nariai Target.
    """
    print("\n" + "="*40)
    print("APERIODIC TILING RATIO ANALYSIS")
    print("="*40)
    
    # Calculate ratios relative to the Nariai attractors
    ratios = [p / f_target for p in peaks]
    
    # The 'Spectre' ratios often involve sqrt(3) or the Golden Ratio
    phi = (1 + 5**0.5) / 2
    sqrt3 = 3**0.5
    
    for i, r in enumerate(ratios):
        print(f"Peak {i+1:02d}: Ratio {r:.4f}", end=" ")
        # Check for proximity to aperiodic constants
        if abs(r - phi) < 0.05: print(" (~Golden Ratio)")
        elif abs(r - sqrt3) < 0.05: print(" (~Hexagonal Symmetry)")
        elif abs(r - 1.0) < 0.05: print(" (~NARIAI LOCK)")
        else: print("")
    print("="*40)

# Observed peaks from your telemetry
observed_peaks = [306.25, 108.50, 624.25, 777.75, 100.00, 501.75, 647.50, 60.00, 354.50, 229.00]
analyze_tiling_ratios(observed_peaks, 230.61)