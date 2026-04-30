import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from gwpy.timeseries import TimeSeries
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# --- CONFIG ---
SUB_FOLDER = "pulsar_fits_large/Rcvr1_2-VEGAS"
CACHE_DIR = "ligo_cache"
MAX_WORKERS = 4 

def get_dual_ligo_data(start, duration=30):
    cache_l1 = os.path.join(CACHE_DIR, f"L1-{int(start)}.hdf5")
    cache_h1 = os.path.join(CACHE_DIR, f"H1-{int(start)}.hdf5")
    results = {}
    try:
        # L1 logic
        if os.path.exists(cache_l1): data_l1 = TimeSeries.read(cache_l1)
        else: data_l1 = TimeSeries.fetch_open_data('L1', start, start+duration); data_l1.write(cache_l1)
        
        # H1 logic
        if os.path.exists(cache_h1): data_h1 = TimeSeries.read(cache_h1)
        else: data_h1 = TimeSeries.fetch_open_data('H1', start, start+duration); data_h1.write(cache_h1)
        
        l1_val = np.std(data_l1.detrend().value)
        h1_val = np.std(data_h1.detrend().value)
        
        # STRIKE NANs AT THE SOURCE
        if np.isnan(l1_val) or np.isnan(h1_val): return None
        return {'L1': l1_val, 'H1': h1_val}
    except: return None

def main():
    print("\n" + "="*60)
    print("      SPECTRE VACUUM CORRELATOR: CALIBRATED")
    print("="*60)
    
    fits_files = sorted([f for f in os.listdir(SUB_FOLDER) if f.endswith('.fits')])
    raw_data = []

    for f in fits_files:
        try:
            with fits.open(os.path.join(SUB_FOLDER, f)) as h:
                hdr = h[0].header
                gps = (hdr['STT_IMJD'] - 44244.0) * 86400 + hdr['STT_SMJD'] - 32 + h['SUBINT'].data['OFFS_SUB'][0]
                
                # FIX: Take the Absolute Value and Mean of the pulsar data 
                # to handle the negative raw values in the VEGAS archive.
                psr_raw = h['SUBINT'].data['DATA']
                psr_intensity = np.abs(np.mean(psr_raw)) 
                
                raw_data.append({'gps': gps, 'psr': psr_intensity})
        except: continue

    print(f"[STAGE 1] Found {len(raw_data)} snaps. Syncing with LIGO...")

    final_triplets = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(get_dual_ligo_data, p['gps']): p for p in raw_data}
        for future in as_completed(future_map):
            p = future_map[future]
            ligo = future.result()
            if ligo:
                final_triplets.append({'gps': p['gps'], 'psr': p['psr'], 'l1': ligo['L1'], 'h1': ligo['H1']})
                print(f" [MATCH] GPS: {p['gps']:.1f} | PSR: {p['psr']:.2e} | L1: {ligo['L1']:.2e} | H1: {ligo['H1']:.2e}")

    if not final_triplets:
        print("[FAIL] No valid triplets survived.")
        return

    final_triplets.sort(key=lambda x: x['gps'])
    
    # Extract arrays
    psr = np.array([d['psr'] for d in final_triplets])
    l1  = np.array([d['l1'] for d in final_triplets])
    h1  = np.array([d['h1'] for d in final_triplets])

    # Standard Correlation
    r_l1 = np.corrcoef(psr, l1)[0,1]
    r_h1 = np.corrcoef(psr, h1)[0,1]

    print("\n" + "="*60)
    print(f"FINAL PHYSICS REPORT (N={len(final_triplets)})")
    print("="*60)
    print(f"Pulsar-Livingston Sync: {r_l1:.6f}")
    print(f"Pulsar-Hanford Sync:    {r_h1:.6f}")
    print(f"Cross-Detector Sync:    {np.corrcoef(l1, h1)[0,1]:.6f}")
    print("="*60)

    plt.figure(figsize=(10,5))
    plt.plot(psr / np.max(psr), 'o-', label="Pulsar (Abs Intensity)", color='gold')
    plt.plot(l1 / np.max(l1), 'r--', label="L1 Vacuum Jitter")
    plt.plot(h1 / np.max(h1), 'b--', label="H1 Vacuum Jitter")
    plt.title("Non-Local Vacuum Synchronization (Calibrated)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()