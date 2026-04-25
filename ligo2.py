import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from scipy.stats import entropy
from scipy.signal import butter, filtfilt

def get_ligo_data(detector, start, end, cache_dir="ligo_cache"):
    if not os.path.exists(cache_dir): os.makedirs(cache_dir)
    cache_file = os.path.join(cache_dir, f"{detector}_{start}_{end}.h5")
    if os.path.exists(cache_file):
        return TimeSeries.read(cache_file)
    data = TimeSeries.fetch_open_data(detector, start, end, sample_rate=4096)
    data.write(cache_file, overwrite=True)
    return data

def manual_distinction_filter(data, low=20, high=1200, fs=4096):
    """Explicitly applies the distinction drive using a Butterworth manifold."""
    nyq = 0.5 * fs
    # Using 1200Hz as a very safe 'Explicate' boundary
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    filtered_data = filtfilt(b, a, data.value)
    return TimeSeries(filtered_data, sample_rate=fs, t0=data.t0)

def analyze_aperiodic_archive(gps_start, gps_end, r_target):
    c = 299792458
    f_jitter_target = c / r_target
    fs = 4096
    
    print("-" * 30)
    print("INITIALIZING APERIODIC ANALYSIS (v1.5)")
    print(f"Target Nariai Radius: {r_target:.2e} m")
    print("-" * 30)

    try:
        # 1. Fetch & Pre-process
        h1 = get_ligo_data('H1', gps_start, gps_end).resample(fs)
        l1 = get_ligo_data('L1', gps_start, gps_end).resample(fs)

        # 2. Apply Manual Filter (The 'Distinction' Step)
        print("Applying Manual Butterworth Distinction Filter...")
        h1_filt = manual_distinction_filter(h1.whiten(), low=20, high=1200, fs=fs)
        l1_filt = manual_distinction_filter(l1.whiten(), low=20, high=1200, fs=fs)

        # 3. Cross-Spectral Density
        csd = h1_filt.csd(l1_filt, fftlength=4, overlap=2)
        csd_mag = np.abs(csd.value)
        
        peak_idx = np.argmax(csd_mag)
        peak_freq = csd.frequencies.value[peak_idx]
        peak_val = csd_mag[peak_idx]

        # 4. Spectral Entropy (Perfect Glass Test)
        psd_h1 = h1_filt.psd(fftlength=4)
        norm_psd = psd_h1.value / np.sum(psd_h1.value)
        aperiodic_entropy = entropy(norm_psd)

        # 5. Output Results
        print("\n" + "="*40)
        print("APERIODIC SIGNATURE RESULTS")
        print("="*40)
        print(f"Aperiodic Entropy:  {aperiodic_entropy:.6f} bits")
        print(f"Target Jitter:      {f_jitter_target:.2f} Hz")
        print(f"Strongest Peak:     {peak_freq:.2f} Hz")
        print(f"Correlation Power:  {peak_val:.2e}")
        print("-" * 40)
        
        # Diagnostic: Raw CSD sample
        print("CSD Snapshot (First 5 bins):")
        print(csd_mag[:5])
        print("="*40 + "\n")

        plt.figure(figsize=(10, 5))
        plt.loglog(csd.frequencies, csd_mag, color='teal')
        plt.axvline(f_jitter_target, color='red', linestyle='--', label='Target')
        plt.title("Path Identity Correlation Magnitude")
        plt.show()

    except Exception as e:
        print(f"ANALYSIS ERROR: {e}")

if __name__ == "__main__":
    analyze_aperiodic_archive(1266624018, 1266624082, 1.3e6)