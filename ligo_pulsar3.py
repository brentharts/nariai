import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from gwpy.timeseries import TimeSeries
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from math import factorial

# --- CONFIG ---
SUB_FOLDER = "pulsar_fits_large/Rcvr1_2-VEGAS"
CACHE_DIR = "ligo_cache"
MAX_WORKERS = 4 

def permutation_entropy(time_series, order=3, delay=1):
    """
    Calculates Permutation Entropy as a proxy for local vacuum entropy.
    Higher values = higher chaos/entropy density.
    """
    x = np.array(time_series)
    hashmult = np.power(order, np.arange(order))
    sorted_idx = np.argsort(np.lib.stride_tricks.sliding_window_view(x, order)[::delay], axis=1)
    hashvalues = (sorted_idx * hashmult).sum(1)
    _, counts = np.unique(hashvalues, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs)) / np.log2(factorial(order))

def get_vacuum_entropy(start, duration=30):
    cache_l1 = os.path.join(CACHE_DIR, f"L1-{int(start)}.hdf5")
    cache_h1 = os.path.join(CACHE_DIR, f"H1-{int(start)}.hdf5")
    try:
        # Load L1
        if os.path.exists(cache_l1): data_l1 = TimeSeries.read(cache_l1)
        else: data_l1 = TimeSeries.fetch_open_data('L1', start, start+duration); data_l1.write(cache_l1)
        
        # Load H1
        if os.path.exists(cache_h1): data_h1 = TimeSeries.read(cache_h1)
        else: data_h1 = TimeSeries.fetch_open_data('H1', start, start+duration); data_h1.write(cache_h1)
        
        # Calculate Entropy for both detectors
        # We use order=3 for quick structural complexity mapping
        ent_l1 = permutation_entropy(data_l1.value, order=3)
        ent_h1 = permutation_entropy(data_h1.value, order=3)
        
        return {'L1_Ent': ent_l1, 'H1_Ent': ent_h1}
    except: return None

def main():
    print("\n" + "="*60)
    print("      ENTROPY-WAVE COUPLING ANALYSIS (NON-LOCAL)")
    print("="*60)
    
    fits_files = sorted([f for f in os.listdir(SUB_FOLDER) if f.endswith('.fits')])
    raw_data = []

    for f in fits_files:
        try:
            with fits.open(os.path.join(SUB_FOLDER, f)) as h:
                hdr = h[0].header
                gps = (hdr['STT_IMJD'] - 44244.0) * 86400 + hdr['STT_SMJD'] - 32 + h['SUBINT'].data['OFFS_SUB'][0]
                # Normalize pulsar intensity to its own local entropy proxy
                psr_raw = np.abs(h['SUBINT'].data['DATA'].flatten())
                psr_ent = permutation_entropy(psr_raw, order=3)
                raw_data.append({'gps': gps, 'psr_ent': psr_ent})
        except: continue

    final_triplets = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(get_vacuum_entropy, p['gps']): p for p in raw_data}
        for future in as_completed(future_map):
            p = future_map[future]
            ent_data = future.result()
            if ent_data:
                final_triplets.append({
                    'gps': p['gps'], 
                    'psr': p['psr_ent'], 
                    'l1': ent_data['L1_Ent'], 
                    'h1': ent_data['H1_Ent']
                })
                print(f" [ENTROPY SYNC] GPS: {p['gps']:.1f} | PSR_S: {p['psr_ent']:.4f} | L1_S: {ent_data['L1_Ent']:.4f} | H1_S: {ent_data['H1_Ent']:.4f}")

    final_triplets.sort(key=lambda x: x['gps'])
    psr = np.array([d['psr'] for d in final_triplets])
    l1  = np.array([d['l1'] for d in final_triplets])
    h1  = np.array([d['h1'] for d in final_triplets])

    print("\n" + "="*60)
    print(f"NON-LOCAL ENTROPY REPORT (N={len(final_triplets)})")
    print("="*60)
    print(f"Pulsar <-> L1 Entropy Correlation: {np.corrcoef(psr, l1)[0,1]:.6f}")
    print(f"Pulsar <-> H1 Entropy Correlation: {np.corrcoef(psr, h1)[0,1]:.6f}")
    print(f"Detector Entropy Cross-Talk:      {np.corrcoef(l1, h1)[0,1]:.6f}")
    print("="*60)

    # Visualization of the Entropy Flow
    plt.figure(figsize=(12, 6))
    plt.plot(psr, 'gold', label="Pulsar Entropy Score", linewidth=2)
    plt.plot(l1, 'r--', label="L1 Vacuum Entropy", alpha=0.7)
    plt.plot(h1, 'b--', label="H1 Vacuum Entropy", alpha=0.7)
    plt.title("Aperiodic Entropy Wave Synchronization")
    plt.ylabel("Permutation Entropy (Normalized)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()