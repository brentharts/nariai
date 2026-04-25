import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from scipy.stats import entropy

def robust_entropy_coupling(start_gps, total_duration):
    """
    Cleans the archive glitches to measure the underlying 
    coupling of informational uniqueness between H1 and L1.
    """
    fs = 4096
    slice_size = 64
    
    print(f"\n--- INITIATING ROBUST ENTROPY COUPLING ({total_duration}s) ---")
    
    h1_full = TimeSeries.read(f"ligo_cache/H1_{start_gps}_{start_gps+total_duration}.h5")
    l1_full = TimeSeries.read(f"ligo_cache/L1_{start_gps}_{start_gps+total_duration}.h5")
    
    h1_ents = []
    l1_ents = []
    valid_times = []
    
    for t in range(0, total_duration - slice_size, slice_size):
        try:
            h_crop = h1_full.crop(start_gps + t, start_gps + t + slice_size).whiten()
            l_crop = l1_full.crop(start_gps + t, start_gps + t + slice_size).whiten()
            
            # Check for data dropouts (zeros or NaNs in the raw data)
            if np.any(np.isnan(h_crop.value)) or np.all(h_crop.value == 0):
                continue
                
            h_psd = h_crop.psd(fftlength=4)
            l_psd = l_crop.psd(fftlength=4)
            
            h_ent = entropy(h_psd.value / np.sum(h_psd.value))
            l_ent = entropy(l_psd.value / np.sum(l_psd.value))
            
            if np.isfinite(h_ent) and np.isfinite(l_ent):
                h1_ents.append(h_ent)
                l1_ents.append(l_ent)
                valid_times.append(t)
        except:
            continue

    # Calculate Cleaned Correlation
    if len(h1_ents) > 1:
        ent_correlation = np.corrcoef(h1_ents, l1_ents)[0, 1]
    else:
        ent_correlation = 0

    print("\n" + "="*40)
    print("CLEANED ENTROPIC COUPLING RESULTS")
    print("="*40)
    print(f"Valid Slices:        {len(h1_ents)}")
    print(f"Entropy Correlation: {ent_correlation:.4f}")
    
    if ent_correlation > 0.4:
        print("STATUS: SHARED DISTINCTION DRIVE (UNIVERSAL)")
    elif ent_correlation > 0.1:
        print("STATUS: WEAK COUPLING (PATH IDENTITY)")
    else:
        print("STATUS: INDEPENDENT MANIFOLDS")
    print("="*40)

    # Plot the 'Breathing' of the Archive
    plt.figure(figsize=(10, 5))
    plt.plot(valid_times, h1_ents, label='H1 Entropy', color='teal', alpha=0.8)
    plt.plot(valid_times, l1_ents, label='L1 Entropy', color='orchid', alpha=0.8)
    plt.title("Universal Archive 'Breathing': Entropic Fluctuations")
    plt.ylabel("Aperiodic Entropy (bits)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

if __name__ == "__main__":
    robust_entropy_coupling(1266624018, 2048)