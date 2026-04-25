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
    white = data.whiten()
    filtered_data = filtfilt(b, a, white.value)
    return TimeSeries(filtered_data, sample_rate=fs, t0=data.t0)

def full_telemetry_scan(start_gps, duration, slice_size=32, r_target=1.3e6):
    c = 299792458
    f_target = c / r_target
    fs = 4096
    
    results = []

    print(f"\n--- INITIATING FULL ARCHIVE TELEMETRY ({duration}s) ---")
    h1_full = get_ligo_data('H1', start_gps, start_gps + duration)
    l1_full = get_ligo_data('L1', start_gps, start_gps + duration)

    for t in range(0, duration - slice_size, slice_size):
        try:
            h1_s = h1_full.crop(start_gps + t, start_gps + t + slice_size)
            l1_s = l1_full.crop(start_gps + t, start_gps + t + slice_size)

            h1_f = manual_distinction_filter(h1_s)
            l1_f = manual_distinction_filter(l1_s)

            csd = h1_f.csd(l1_f, fftlength=4, overlap=2)
            psd = h1_f.psd(fftlength=4)
            
            ent = entropy(psd.value / np.sum(psd.value))
            peak_f = csd.frequencies.value[np.argmax(np.abs(csd.value))]
            
            if np.isfinite(ent) and peak_f > 0:
                is_locked = abs(peak_f - f_target) < (0.02 * f_target)
                status = "[LOCKED]" if is_locked else "        "
                print(f"T+{t:03d}s | {status} Freq: {peak_f:7.2f} Hz | Entropy: {ent:.6f}")
                results.append({'t': t, 'f': peak_f, 'e': ent, 'locked': is_locked})
        except Exception:
            continue

    # Summary Statistics
    entropies = [r['e'] for r in results]
    peaks = [r['f'] for r in results]
    lock_count = sum(1 for r in results if r['locked'])
    
    print("\n" + "="*40)
    print("FINAL ARCHIVE TELEMETRY SUMMARY")
    print("="*40)
    print(f"Global Mean Entropy: {np.mean(entropies):.6f} bits")
    print(f"Total Slices:        {len(results)}")
    print(f"Nariai Lock-ins:     {lock_count}")
    print(f"Lock-in Rate:        {(lock_count/len(results))*100:.2f}%")
    print("="*40 + "\n")

    # Plotting Logic
    plt.figure(figsize=(10, 6))
    plt.scatter([r['t'] for r in results], peaks, c=entropies, cmap='magma', s=100, edgecolors='black')
    plt.axhline(f_target, color='cyan', linestyle='--', label=f'Nariai Target ({f_target:.1f}Hz)')
    plt.title("Aperiodic Jitter Telemetry: Frequency vs. Time")
    plt.xlabel("Time Offset (s)")
    plt.ylabel("Peak Correlation Frequency (Hz)")
    plt.colorbar(label="Aperiodic Entropy (bits)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    robust_analysis_gps = 1266624018 
    full_telemetry_scan(robust_analysis_gps, 512, slice_size=32, r_target=1.3e6)