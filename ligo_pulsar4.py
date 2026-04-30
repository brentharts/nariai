import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from gwpy.timeseries import TimeSeries
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_selection import mutual_info_regression
from math import factorial

# --- CONFIG ---
SUB_FOLDER = "pulsar_fits_large/Rcvr1_2-VEGAS"
CACHE_DIR = "ligo_cache"
MAX_WORKERS = 4 

def permutation_entropy(time_series, order=3, delay=1):
    """Measures complexity/entropy density of the vacuum signal."""
    x = np.array(time_series)
    if len(x) < order: return 0.0
    # Sliding window view for permutation patterns
    sw = np.lib.stride_tricks.sliding_window_view(x, order)[::delay]
    sorted_idx = np.argsort(sw, axis=1)
    # Hash the patterns to find unique "states"
    hashmult = np.power(order, np.arange(order))
    hashvalues = (sorted_idx * hashmult).sum(1)
    _, counts = np.unique(hashvalues, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs)) / np.log2(factorial(order))

def get_vacuum_data(start, duration=30):
    """Retrieves dual-detector data and computes entropy proxies."""
    try:
        data = {}
        for det in ['L1', 'H1']:
            cache_path = os.path.join(CACHE_DIR, f"{det}-{int(start)}.hdf5")
            if os.path.exists(cache_path):
                ts = TimeSeries.read(cache_path)
            else:
                ts = TimeSeries.fetch_open_data(det, start, start+duration)
                ts.write(cache_path)
            
            # Use detrended values to focus on high-frequency vacuum 'jitter'
            clean_ts = ts.detrend().value
            ent = permutation_entropy(clean_ts, order=3)
            
            # Avoid the 'flat-line' 0.0 entropy glitches found in previous runs
            if ent < 0.01: return None 
            data[det] = ent
        return data
    except: return None

def main():
    print("\n" + "="*70)
    print("      NON-LOCAL VACUUM SYNC: MUTUAL INFORMATION (MI) ANALYSIS")
    print("="*70)
    
    fits_files = sorted([f for f in os.listdir(SUB_FOLDER) if f.endswith('.fits')])
    raw_points = []

    for f in fits_files:
        try:
            with fits.open(os.path.join(SUB_FOLDER, f)) as h:
                hdr = h[0].header
                gps = (hdr['STT_IMJD'] - 44244.0) * 86400 + hdr['STT_SMJD'] - 32 + h['SUBINT'].data['OFFS_SUB'][0]
                psr_data = np.abs(h['SUBINT'].data['DATA'].flatten())
                psr_ent = permutation_entropy(psr_data, order=3)
                if psr_ent > 0.01:
                    raw_points.append({'gps': gps, 'psr_s': psr_ent})
        except: continue

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(get_vacuum_data, p['gps']): p for p in raw_points}
        for future in as_completed(future_map):
            p = future_map[future]
            vac = future.result()
            if vac:
                results.append({'gps': p['gps'], 'psr': p['psr_s'], 'l1': vac['L1'], 'h1': vac['H1']})
                print(f" [DATA] GPS {p['gps']:.1f} | PSR_S: {p['psr_s']:.4f} | L1_S: {vac['L1']:.4f} | H1_S: {vac['H1']:.4f}")

    if len(results) < 5:
        print("[!] Not enough data points for MI calculation.")
        return

    results.sort(key=lambda x: x['gps'])
    psr = np.array([r['psr'] for r in results]).reshape(-1, 1)
    l1 = np.array([r['l1'] for r in results])
    h1 = np.array([r['h1'] for r in results])

    # MI Calculation: This is the "Shared Information" in Bits
    mi_l1 = mutual_info_regression(psr, l1, random_state=42)[0]
    mi_h1 = mutual_info_regression(psr, h1, random_state=42)[0]
    mi_detectors = mutual_info_regression(l1.reshape(-1, 1), h1, random_state=42)[0]

    print("\n" + "="*70)
    print(f"INFORMATION THEORETIC REPORT (N={len(results)})")
    print("="*70)
    print(f"Shared Info (Pulsar <-> L1): {mi_l1:.6f} bits")
    print(f"Shared Info (Pulsar <-> H1): {mi_h1:.6f} bits")
    print(f"Shared Info (L1 <-> H1):     {mi_detectors:.6f} bits")
    print("-" * 70)
    print("INTERPRETATION:")
    print(" * MI > 0.1: Significant non-linear coupling detected.")
    print(" * MI > 0.3: Strong evidence of non-local information exchange.")
    print("="*70)

    # Plotting code remains for visual confirmation
    plt.figure(figsize=(10, 5))
    plt.scatter(psr, l1, color='red', label='L1 Entropy')
    plt.scatter(psr, h1, color='blue', label='H1 Entropy')
    plt.xlabel("Pulsar Entropy (Source)")
    plt.ylabel("Vacuum Entropy (Detector)")
    plt.title("Phase-Space Mapping of Non-Local Sync")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()