import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from scipy.signal import welch, coherence

def get_ligo_data(detector, start, end):
    # Utilizing your existing local cache logic
    import os
    cache_file = f"ligo_cache/{detector}_{start}_{end}.h5"
    if os.path.exists(cache_file):
        return TimeSeries.read(cache_file)
    return TimeSeries.fetch_open_data(detector, start, end, sample_rate=4096)

def analyze_phase_coherence(start_gps, target_offset, r_target=1.3e6):
    """
    Measures the Magnitude Squared Coherence between H1 and L1
    specifically for the sub-structure cluster.
    """
    fs = 4096
    t_start = start_gps + target_offset
    t_end = t_start + 32
    
    print(f"\n--- MEASURING MANIFOLD COHERENCE (T+{target_offset}s) ---")
    
    # Load and Filter
    h1 = get_ligo_data('H1', start_gps, start_gps + 512).crop(t_start, t_end).whiten()
    l1 = get_ligo_data('L1', start_gps, start_gps + 512).crop(t_start, t_end).whiten()

    # Calculate Coherence (The 'Path Identity' check)
    freqs, coh = coherence(h1.value, l1.value, fs=fs, nperseg=fs) # 1Hz resolution

    # Focus on the Nariai Band
    mask = (freqs > 150) & (freqs < 350)
    band_freqs = freqs[mask]
    band_coh = coh[mask]
    
    peak_coh_idx = np.argmax(band_coh)
    
    print("\n" + "="*40)
    print("PATH IDENTITY COHERENCE RESULTS")
    print("="*40)
    print(f"Peak Coherence: {band_coh[peak_coh_idx]:.4f}")
    print(f"At Frequency:   {band_freqs[peak_coh_idx]:.2f} Hz")
    print(f"Mean Band Coh:  {np.mean(band_coh):.4f}")
    print("="*40)

    plt.figure(figsize=(10, 4))
    plt.plot(band_freqs, band_coh, color='crimson', label='H1-L1 Coherence')
    plt.axvline(230.61, color='black', linestyle='--', label='Theoretical Nariai')
    plt.title("Manifold Coherence: Path Identity Verification")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence (0-1)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    analyze_phase_coherence(1266624018, 448)