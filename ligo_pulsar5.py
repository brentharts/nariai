import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from gwpy.timeseries import TimeSeries
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_selection import mutual_info_regression
from math import factorial

# --- CONFIGURATION ---
SUB_FOLDER = "pulsar_fits_large/Rcvr1_2-VEGAS"
CACHE_DIR = "ligo_cache"
MAX_WORKERS = 4 
ROLLING_WINDOW = 8 # Number of snapshots to calculate shared information over

def permutation_entropy(time_series, order=3, delay=1):
    """Calculates Permutation Entropy (Proxy for Vacuum Complexity)."""
    x = np.array(time_series)
    if len(x) < order: return 0.0
    sw = np.lib.stride_tricks.sliding_window_view(x, order)[::delay]
    sorted_idx = np.argsort(sw, axis=1)
    hashmult = np.power(order, np.arange(order))
    hashvalues = (sorted_idx * hashmult).sum(1)
    _, counts = np.unique(hashvalues, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs)) / np.log2(factorial(order))

def get_vacuum_entropy_pair(start, duration=30):
    """Fetches L1 and H1 data, cleans it, and returns entropy scores."""
    try:
        data = {}
        for det in ['L1', 'H1']:
            cache_path = os.path.join(CACHE_DIR, f"{det}-{int(start)}.hdf5")
            if os.path.exists(cache_path):
                ts = TimeSeries.read(cache_path)
            else:
                ts = TimeSeries.fetch_open_data(det, start, start+duration)
                ts.write(cache_path)
            
            # Detrending is critical to isolate the aperiodic vacuum jitter
            val = ts.detrend().value
            ent = permutation_entropy(val, order=3)
            
            # Filter out flat-line/glitch segments
            if ent < 0.05: return None 
            data[det] = ent
        return data
    except Exception as e:
        return None

def main():
    print("\n" + "="*80)
    print("      SPECTRE BRIDGE ANALYSIS: UNIFIED NON-LOCAL ENTROPY FLOW")
    print("="*80)
    
    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
    
    fits_files = sorted([f for f in os.listdir(SUB_FOLDER) if f.endswith('.fits')])
    print(f"[DEBUG] Found {len(fits_files)} pulsar FITS files. Harvesting timestamps...")

    raw_list = []
    for f in fits_files:
        try:
            with fits.open(os.path.join(SUB_FOLDER, f)) as h:
                hdr = h[0].header
                # Calculate precision GPS
                gps = (hdr['STT_IMJD'] - 44244.0) * 86400 + hdr['STT_SMJD'] - 32 + h['SUBINT'].data['OFFS_SUB'][0]
                
                # Capture both Entropy (S) and Raw Intensity (Energy Density)
                psr_raw = np.abs(h['SUBINT'].data['DATA'].flatten())
                psr_ent = permutation_entropy(psr_raw, order=3)
                psr_intensity = np.mean(psr_raw)
                
                if psr_ent > 0.05:
                    raw_list.append({'gps': gps, 'psr_ent': psr_ent, 'psr_flux': psr_intensity})
        except: continue

    print(f"[DEBUG] Snapshots ready: {len(raw_list)}. Syncing with LIGO detectors...")

    final_data = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_gps = {executor.submit(get_vacuum_entropy_pair, p['gps']): p for p in raw_list}
        for future in as_completed(future_to_gps):
            p = future_to_gps[future]
            vac = future.result()
            if vac:
                final_data.append({
                    'gps': p['gps'], 
                    'psr_s': p['psr_ent'], 
                    'psr_f': p['psr_flux'],
                    'l1_s': vac['L1'], 
                    'h1_s': vac['H1']
                })
                print(f" [SYNCED] GPS: {p['gps']:.1f} | PSR Flux: {p['psr_flux']:.2e} | L1_S: {vac['L1']:.4f} | H1_S: {vac['H1']:.4f}")

    if len(final_data) < ROLLING_WINDOW:
        print("[FAIL] Insufficient synced data points for rolling analysis.")
        return

    # Sort by time for rolling window
    final_data.sort(key=lambda x: x['gps'])
    
    gps_arr = np.array([d['gps'] for d in final_data])
    psr_f   = np.array([d['psr_f'] for d in final_data])
    l1_s    = np.array([d['l1_s'] for d in final_data])
    h1_s    = np.array([d['h1_s'] for d in final_data])

    # --- ROLLING MUTUAL INFORMATION ENGINE ---
    print(f"\n[DEBUG] Calculating Rolling Bridge (Window={ROLLING_WINDOW})...")
    
    steps = len(final_data) - ROLLING_WINDOW + 1
    bridge_mi = []
    energy_flux = []
    timeline = []

    for i in range(steps):
        win_l1 = l1_s[i : i + ROLLING_WINDOW].reshape(-1, 1)
        win_h1 = h1_s[i : i + ROLLING_WINDOW]
        win_psr = psr_f[i : i + ROLLING_WINDOW]
        
        # Cross-Detector Bridge Strength (MI)
        mi_score = mutual_info_regression(win_l1, win_h1, random_state=42)[0]
        
        bridge_mi.append(mi_score)
        energy_flux.append(np.mean(win_psr))
        timeline.append(gps_arr[i + ROLLING_WINDOW // 2])

    # Final Coupling Calculation
    coupling = np.corrcoef(energy_flux, bridge_mi)[0, 1]

    print("\n" + "="*80)
    print("      FINAL PHYSICS REPORT: ENERGY-BRIDGE COUPLING")
    print("="*80)
    print(f"Detectors:            Livingston (L1) & Hanford (H1)")
    print(f"Synchronization:      Entropy-Wave Mutual Information")
    print(f"Total Sync Points:    {len(final_data)}")
    print(f"Rolling Correlation:  {coupling:.6f}")
    print("-" * 80)
    if coupling > 0.4:
        print("INTERPRETATION: High Coupling. Energy density reinforces the aperiodic bridge.")
    else:
        print("INTERPRETATION: Low Coupling. The bridge exists independently of local flux.")
    print("="*80)

    # VISUALIZATION
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('GPS Time')
    ax1.set_ylabel('Pulsar Energy Flux (Energy Density)', color='gold')
    ax1.plot(timeline, energy_flux, color='gold', linewidth=3, label='Pulsar Flux')
    ax1.tick_params(axis='y', labelcolor='gold')

    ax2 = ax1.twinx()
    ax2.set_ylabel('L1-H1 Shared Info Bridge (Bits)', color='cyan')
    ax2.plot(timeline, bridge_mi, color='cyan', linestyle='--', marker='o', label='Information Bridge')
    ax2.tick_params(axis='y', labelcolor='cyan')

    plt.title("The Spectre Mechanism: Do Energy Waves Build Information Bridges?")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()