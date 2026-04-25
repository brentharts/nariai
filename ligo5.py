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
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    # Whiten to reveal the underlying aperiodic noise floor
    white = data.whiten()
    filtered_data = filtfilt(b, a, white.value)
    return TimeSeries(filtered_data, sample_rate=fs, t0=data.t0)

def robust_archive_analysis(start_gps, duration, slice_size=32, r_target=1.3e6):
    c = 299792458
    f_target = c / r_target
    fs = 4096
    
    entropies = []
    peaks = []
    time_indices = []

    print(f"\n--- ANALYZING ARCHIVE DISTRIBUTION ({duration}s) ---")
    h1_full = get_ligo_data('H1', start_gps, start_gps + duration)
    l1_full = get_ligo_data('L1', start_gps, start_gps + duration)

    for t in range(0, duration - slice_size, slice_size):
        try:
            h1_s = h1_full.crop(start_gps + t, start_gps + t + slice_size)
            l1_s = l1_full.crop(start_gps + t, start_gps + t + slice_size)

            h1_f = manual_distinction_filter(h1_s)
            l1_f = manual_distinction_filter(l1_s)

            # Measure CSD and Entropy
            csd = h1_f.csd(l1_f, fftlength=4, overlap=2)
            csd_mag = np.abs(csd.value)
            psd = h1_f.psd(fftlength=4)
            
            ent = entropy(psd.value / np.sum(psd.value))
            peak_f = csd.frequencies.value[np.argmax(csd_mag)]
            
            # Only record finite, non-zero data
            if np.isfinite(ent) and peak_f > 0:
                entropies.append(ent)
                peaks.append(peak_f)
                time_indices.append(t)
        except Exception:
            continue

    # --- Results Calculation ---
    mean_ent = np.mean(entropies)
    # Lock-in: frequency within 2% of target
    locked = [p for p in peaks if abs(p - f_target) < (0.02 * f_target)]
    lock_rate = (len(locked) / len(peaks)) * 100 if peaks else 0

    print("\n" + "="*40)
    print("CLEANED APERIODIC RESULTS")
    print("="*40)
    print(f"Mean Entropy:      {mean_ent:.6f} bits")
    print(f"Nariai Target:     {f_target:.2f} Hz")
    print(f"Lock-in Rate:      {lock_rate:.1f}%")
    print(f"Peak Count:        {len(peaks)} slices")
    print("="*40 + "\n")

    # Visualization: Persistence Histogram
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(peaks, bins=25, color='teal', edgecolor='black', alpha=0.7)
    plt.axvline(f_target, color='red', linestyle='--', label=f'Nariai Limit ({f_target:.1f}Hz)')
    plt.title("Distribution of Jitter Peaks")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Detections")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(time_indices, peaks, c=entropies, cmap='viridis', s=50)
    plt.axhline(f_target, color='red', linestyle='--')
    plt.colorbar(label='Aperiodic Entropy')
    plt.title("Jitter Frequency vs. Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Peak Frequency (Hz)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    robust_archive_analysis(1266624018, 512, slice_size=32, r_target=1.3e6)