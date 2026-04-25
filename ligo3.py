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
    # Whiten first to flatten the 'Explicate' manifold
    white = data.whiten()
    filtered_data = filtfilt(b, a, white.value)
    return TimeSeries(filtered_data, sample_rate=fs, t0=data.t0)

def scan_temporal_archive(start_gps, duration, slice_size=16, r_target=1.3e6):
    c = 299792458
    f_target = c / r_target
    fs = 4096
    
    # Results containers
    times = []
    entropies = []
    peaks = []

    print(f"SCANNING ARCHIVE: {duration}s window in {slice_size}s increments")
    
    # Pre-fetch large blocks to cache
    h1_full = get_ligo_data('H1', start_gps, start_gps + duration)
    l1_full = get_ligo_data('L1', start_gps, start_gps + duration)

    for t in range(0, duration - slice_size, slice_size):
        try:
            t_start = start_gps + t
            t_end = t_start + slice_size
            
            # Slice the data manifold
            h1_s = h1_full.crop(t_start, t_end)
            l1_s = l1_full.crop(t_start, t_end)

            # Process
            h1_f = manual_distinction_filter(h1_s, fs=fs)
            l1_f = manual_distinction_filter(l1_s, fs=fs)

            # CSD and Entropy
            csd = h1_f.csd(l1_f, fftlength=4, overlap=2)
            psd = h1_f.psd(fftlength=4)
            
            ent = entropy(psd.value / np.sum(psd.value))
            peak_f = csd.frequencies.value[np.argmax(np.abs(csd.value))]
            
            times.append(t)
            entropies.append(ent)
            peaks.append(peak_f)
            
            print(f"T+{t:02d}s | Entropy: {ent:.4f} | Peak: {peak_f:.2f} Hz")
            
        except Exception as e:
            continue

    # Final Summary for copy-paste
    print("\n" + "="*40)
    print("TEMPORAL DRIFT SUMMARY")
    print("="*40)
    print(f"Mean Entropy: {np.mean(entropies):.6f}")
    print(f"Peak Variance: {np.var(peaks):.2f}")
    print(f"F_Jitter Predicted: {f_target:.2f} Hz")
    print("="*40)

if __name__ == "__main__":
    # Scanning a 64-second window to look for "Archive Persistence"
    scan_temporal_archive(1266624018, 64, slice_size=8)