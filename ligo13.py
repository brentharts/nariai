import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

def non_local_jitter_correlation(start_gps, total_duration, target_f=513.9):
    """
    Compares the jitter patterns between H1 and L1 to verify 
    the Non-Local Path Identity.
    """
    fs = 4096
    slice_size = 64
    step = slice_size // 4
    
    print(f"\n--- VERIFYING NON-LOCAL JITTER CORRELATION ({total_duration}s) ---")
    
    h1_full = TimeSeries.read(f"ligo_cache/H1_{start_gps}_{start_gps+total_duration}.h5")
    l1_full = TimeSeries.read(f"ligo_cache/L1_{start_gps}_{start_gps+total_duration}.h5")
    
    times = []
    h1_freqs = []
    l1_freqs = []
    
    for t in range(0, total_duration - slice_size, step):
        # High-res PSD for both detectors
        h_psd = h1_full.crop(start_gps + t, start_gps + t + slice_size).whiten().psd(fftlength=slice_size)
        l_psd = l1_full.crop(start_gps + t, start_gps + t + slice_size).whiten().psd(fftlength=slice_size)
        
        mask = (h_psd.frequencies.value > target_f - 3) & (h_psd.frequencies.value < target_f + 3)
        
        h1_freqs.append(h_psd.frequencies.value[mask][np.argmax(h_psd.value[mask])])
        l1_freqs.append(l_psd.frequencies.value[mask][np.argmax(l_psd.value[mask])])
        times.append(t)

    # Calculate Pearson Correlation Coefficient of the Jitter
    correlation = np.corrcoef(h1_freqs, l1_freqs)[0, 1]
    
    print("\n" + "="*40)
    print("NON-LOCAL ARCHIVE CORRELATION")
    print("="*40)
    print(f"Jitter Correlation: {correlation:.4f}")
    if correlation > 0.5:
        print("STATUS: UNIVERSAL PATH IDENTITY CONFIRMED")
    else:
        print("STATUS: LOCALIZED FLUCTUATIONS")
    print("="*40)

    plt.figure(figsize=(12, 5))
    plt.plot(times, np.array(h1_freqs) - target_f, label='H1 Jitter', color='cyan', alpha=0.7)
    plt.plot(times, np.array(l1_freqs) - target_f, label='L1 Jitter', color='magenta', alpha=0.7)
    plt.title("Synchronized Aperiodic Jitter (H1 vs L1)")
    plt.ylabel("Delta Freq (Hz)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    non_local_jitter_correlation(1266624018, 2048, target_f=513.9)