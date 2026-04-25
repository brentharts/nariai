import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from scipy.signal import coherence, butter, filtfilt

def get_deep_archive(detector, start, duration):
    cache_file = f"ligo_cache/{detector}_{start}_{start+duration}.h5"
    if os.path.exists(cache_file):
        print(f"Loading {detector} Deep Archive from cache...")
        return TimeSeries.read(cache_file)
    print(f"Downloading {detector} Deep Archive ({duration}s)...")
    data = TimeSeries.fetch_open_data(detector, start, start+duration, sample_rate=4096)
    data.write(cache_file, overwrite=True)
    return data

def deep_manifold_scan(start_gps, total_duration, slice_size=128):
    fs = 4096
    print(f"\n--- INITIATING DEEP ARCHIVE INTEGRATION ({total_duration}s) ---")
    
    h1_big = get_deep_archive('H1', start_gps, total_duration)
    l1_big = get_deep_archive('L1', start_gps, total_duration)

    global_times = []
    global_coherence = []
    global_freqs = []

    # Sliding window with 50% overlap for continuity
    step = slice_size // 2
    for t in range(0, total_duration - slice_size, step):
        try:
            h1_s = h1_big.crop(start_gps + t, start_gps + t + slice_size).whiten()
            l1_s = l1_big.crop(start_gps + t, start_gps + t + slice_size).whiten()

            # High-resolution coherence
            f, c = coherence(h1_s.value, l1_s.value, fs=fs, nperseg=fs*2)
            
            # Focus on the 'Explicate' band (20Hz - 1000Hz)
            mask = (f > 20) & (f < 1000)
            peak_idx = np.argmax(c[mask])
            
            global_times.append(t)
            global_coherence.append(c[mask][peak_idx])
            global_freqs.append(f[mask][peak_idx])
            
            if t % 256 == 0:
                print(f"Progress: {t:4d}s | Max Coh: {global_coherence[-1]:.4f} at {global_freqs[-1]:.2f} Hz")
        except:
            continue

    print("\n" + "="*40)
    print("DEEP ARCHIVE COHERENCE SUMMARY")
    print("="*40)
    print(f"Max Global Coherence: {np.max(global_coherence):.4f}")
    print(f"Frequency at Max:    {global_freqs[np.argmax(global_coherence)]:.2f} Hz")
    print(f"Mean Stability:      {np.mean(global_coherence):.4f}")
    print("="*40)

    # Persistence Plot
    plt.figure(figsize=(12, 5))
    plt.scatter(global_times, global_freqs, c=global_coherence, cmap='inferno', s=30)
    plt.colorbar(label='Coherence Magnitude')
    plt.axhline(230.61, color='cyan', linestyle='--', alpha=0.5, label='Nariai Target')
    plt.title("Deep Archive Path Identity Persistence")
    plt.xlabel("Time Offset (s)")
    plt.ylabel("Coherent Frequency (Hz)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Deep dive into a 2048-second manifold
    deep_manifold_scan(1266624018, 2048)