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
        print(f"Loading {detector} from cache...")
        return TimeSeries.read(cache_file)
    print(f"Cache miss. Downloading {detector} ({end-start}s)...")
    data = TimeSeries.fetch_open_data(detector, start, end, sample_rate=4096)
    data.write(cache_file, overwrite=True)
    return data

def manual_distinction_filter(data, low=20, high=1200, fs=4096):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    white = data.whiten()
    filtered_data = filtfilt(b, a, white.value)
    return TimeSeries(filtered_data, sample_rate=fs, t0=data.t0)

def long_window_archive_scan(start_gps, duration, slice_size=32, r_target=1.3e6):
    c = 299792458
    f_target = c / r_target
    fs = 4096
    
    # Results containers
    entropies = []
    peaks = []

    print(f"\n--- STARTING LONG-WINDOW SCAN ({duration}s) ---")
    
    # 1. Fetch entire manifold
    h1_full = get_ligo_data('H1', start_gps, start_gps + duration)
    l1_full = get_ligo_data('L1', start_gps, start_gps + duration)

    # 2. Iterative Analysis
    for t in range(0, duration - slice_size, slice_size):
        try:
            t_start = start_gps + t
            t_end = t_start + slice_size
            h1_s = h1_full.crop(t_start, t_end)
            l1_s = l1_full.crop(t_start, t_end)

            h1_f = manual_distinction_filter(h1_s)
            l1_f = manual_distinction_filter(l1_s)

            # Cross-Spectral Density
            csd = h1_f.csd(l1_f, fftlength=4, overlap=2)
            csd_mag = np.abs(csd.value)
            
            # Entropy check
            psd = h1_f.psd(fftlength=4)
            ent = entropy(psd.value / np.sum(psd.value))
            
            peak_f = csd.frequencies.value[np.argmax(csd_mag)]
            
            entropies.append(ent)
            peaks.append(peak_f)
            
            print(f"Slice {t:03d}s | Entropy: {ent:.4f} | Peak: {peak_f:.2f} Hz")
        except Exception:
            continue

    # 3. Final Summary Output
    print("\n" + "="*40)
    print("LONG-WINDOW ARCHIVE SUMMARY")
    print("="*40)
    print(f"Total Duration:   {duration}s")
    print(f"Global Mean Ent:  {np.mean(entropies):.6f}")
    print(f"Frequency Drift:  {np.min(peaks):.1f} Hz to {np.max(peaks):.1f} Hz")
    print(f"Jitter Predict:   {f_target:.2f} Hz")
    print("="*40 + "\n")

    # 4. Global Plotting (from last slice as representative)
    plt.figure(figsize=(12, 6))
    plt.loglog(csd.frequencies, csd_mag, color='midnightblue', alpha=0.8, label='Correlation (CSD)')
    plt.axvline(f_target, color='crimson', linestyle='--', label=f'Predicted Nariai Jitter ({f_target:.1f}Hz)')
    plt.title(f"Aperiodic Archive Analysis: {duration}s Window\n(Mean Entropy: {np.mean(entropies):.4f})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Correlation Magnitude")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

if __name__ == "__main__":
    # Expand to a 512-second window for deep archive analysis
    GPS_START = 1266624018 
    DURATION = 512
    R_NARIAI = 1.3e6
    
    long_window_archive_scan(GPS_START, DURATION, slice_size=32, r_target=R_NARIAI)