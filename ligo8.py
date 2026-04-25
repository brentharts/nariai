import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from scipy.signal import butter, filtfilt

def get_ligo_data(detector, start, end):
    # Utilizing your existing local cache logic
    import os
    cache_file = f"ligo_cache/{detector}_{start}_{end}.h5"
    if os.path.exists(cache_file):
        return TimeSeries.read(cache_file)
    return TimeSeries.fetch_open_data(detector, start, end, sample_rate=4096)

def spatial_wavefront_scan(start_gps, target_slice_offset, r_target=1.3e6):
    """
    Performs a high-temporal-resolution scan of a specific 
    locked window to find fine-structure sub-peaks.
    """
    c = 299792458
    f_target = c / r_target
    fs = 4096
    
    # Zoom into the T+448s mark for a 32s window
    t_start = start_gps + target_slice_offset
    t_end = t_start + 32
    
    print(f"\n--- PERFORMING WAVEFRONT RECONSTRUCTION (T+{target_slice_offset}s) ---")
    
    h1 = get_ligo_data('H1', start_gps, start_gps + 512).crop(t_start, t_end)
    l1 = get_ligo_data('L1', start_gps, start_gps + 512).crop(t_start, t_end)

    # Apply the Distinction Filter
    nyq = 0.5 * fs
    b, a = butter(4, [20/nyq, 1200/nyq], btype='band')
    h1_f = TimeSeries(filtfilt(b, a, h1.whiten().value), sample_rate=fs)
    l1_f = TimeSeries(filtfilt(b, a, l1.whiten().value), sample_rate=fs)

    # HIGH RESOLUTION: Small fftlength (1s) to see sub-slice evolution
    # This reveals the 'fine structure' of the bit-threads
    csd = h1_f.csd(l1_f, fftlength=1, overlap=0.5)
    csd_mag = np.abs(csd.value)
    
    # Find top 3 sub-peaks in the vicinity of the Nariai frequency
    mask = (csd.frequencies.value > f_target - 50) & (csd.frequencies.value < f_target + 50)
    local_freqs = csd.frequencies.value[mask]
    local_mags = csd_mag[mask]
    
    # Sort to find the primary sub-structure
    sort_idx = np.argsort(local_mags)[::-1]
    
    print("\n" + "="*40)
    print("SPATIAL WAVEFRONT SUB-STRUCTURE")
    print("="*40)
    print(f"Target Center:  {f_target:.2f} Hz")
    for i in range(min(5, len(sort_idx))):
        f_sub = local_freqs[sort_idx[i]]
        p_sub = local_mags[sort_idx[i]]
        print(f"Sub-Peak {i+1}: {f_sub:7.2f} Hz | Power: {p_sub:.2e}")
    print("="*40 + "\n")

    # Plot the fine structure
    plt.figure(figsize=(10, 5))
    plt.plot(csd.frequencies, csd_mag, color='darkorchid', label='CSD Fine Structure')
    plt.axvline(f_target, color='red', linestyle='--', label='Nariai Center')
    plt.xlim(f_target - 100, f_target + 100)
    plt.title(f"Wavefront Reconstruction: Sub-structure near {f_target:.1f}Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Correlation Magnitude")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Scan the T+448s slice where we saw the Nariai Lock
    spatial_wavefront_scan(1266624018, 448, r_target=1.3e6)