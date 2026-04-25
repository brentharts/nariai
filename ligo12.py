import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

def micro_frequency_telemetry(start_gps, total_duration, target_f=513.50):
    """
    Performs ultra-high resolution frequency tracking of the 
    513.50 Hz attractor to look for aperiodic jitter.
    """
    fs = 4096
    # 64-second slices for 0.015 Hz resolution
    slice_size = 64 
    
    print(f"\n--- ANALYZING FINE-STRUCTURE JITTER (Target: {target_f} Hz) ---")
    
    h1_full = TimeSeries.read(f"ligo_cache/H1_{start_gps}_{start_gps+total_duration}.h5")
    
    times = []
    exact_freqs = []
    
    for t in range(0, total_duration - slice_size, slice_size // 4):
        slice_data = h1_full.crop(start_gps + t, start_gps + t + slice_size).whiten()
        
        # Power Spectral Density with high resolution
        psd = slice_data.psd(fftlength=slice_size)
        
        # Find the peak specifically around our 513.5 Hz attractor
        mask = (psd.frequencies.value > target_f - 2) & (psd.frequencies.value < target_f + 2)
        local_freqs = psd.frequencies.value[mask]
        local_power = psd.value[mask]
        
        peak_f = local_freqs[np.argmax(local_power)]
        
        times.append(t)
        exact_freqs.append(peak_f)
        
    # Calculate the 'Aperiodic Variance'
    jitter = np.std(exact_freqs)
    
    print("\n" + "="*40)
    print("ATTRACTOR JITTER TELEMETRY")
    print("="*40)
    print(f"Mean Frequency: {np.mean(exact_freqs):.4f} Hz")
    print(f"Spectral Jitter: {jitter:.6f} Hz")
    if jitter > 0.001:
        print("STATUS: APERIODIC SIGNAL CONFIRMED")
    else:
        print("STATUS: STATIONARY LINE (POTENTIAL ARTIFACT)")
    print("="*40)

    plt.figure(figsize=(10, 4))
    plt.plot(times, exact_freqs, color='lime', drawstyle='steps-post')
    plt.axhline(target_f, color='white', linestyle='--', alpha=0.3)
    plt.title("Aperiodic Jitter of the 513.5 Hz Attractor")
    plt.ylabel("Exact Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.grid(True, alpha=0.2)
    plt.show()

if __name__ == "__main__":
    micro_frequency_telemetry(1266624018, 2048, target_f=513.50)