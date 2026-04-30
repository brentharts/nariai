# https://zenodo.org/records/7236460
# An unusual pulse shape change event in PSR J1713+0747 observed with the Green Bank Telescope and CHIME: Profile data and figure reproduction scripts
# Jennings, Ross J. et al. (2022)
# wget -c --tries=inf -O psr_large_data.tar.gz https://zenodo.org/records/7236460/files/Rcvr1_2-VEGAS-fits.tar.gz

import os, tarfile
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from gwpy.timeseries import TimeSeries
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# --- CONFIG ---
SUB_FOLDER = "pulsar_fits_large/Rcvr1_2-VEGAS"
CACHE_DIR = "ligo_cache"
MAX_WORKERS = 2 # Balanced for dual-stream downloads

ZENODO_URL = 'https://zenodo.org/records/10622185/files/Rcvr1_2-VEGAS-fits.tar.gz'
LOCAL_TAR = "psr_large_data.tar.gz"
EXTRACT_DIR = "pulsar_fits_large"
# Subfolder inside the tar is likely Rcvr1_2-VEGAS
SUB_FOLDER = os.path.join(EXTRACT_DIR, "Rcvr1_2-VEGAS")

def download_and_extract():
    if not os.path.exists(LOCAL_TAR):
        print(f"[1/3] Downloading Large PSRFITS archive (770MB). This will take a few minutes...")
        # wget -c --tries=inf -O psr_large_data.tar.gz https://zenodo.org/records/10622185/files/Rcvr1_2-VEGAS-fits.tar.gz
        r = requests.get(ZENODO_URL, stream=True)
        with open(LOCAL_TAR, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)
    
    if not os.path.exists(EXTRACT_DIR):
        print(f"[2/3] Extracting archive...")
        with tarfile.open(LOCAL_TAR, "r:gz") as tar:
            tar.extractall(path=EXTRACT_DIR)

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_dual_ligo_data(start, duration=30):
    """Fetches BOTH L1 and H1. Fails if either is unavailable."""
    cache_l1 = os.path.join(CACHE_DIR, f"L1-{int(start)}.hdf5")
    cache_h1 = os.path.join(CACHE_DIR, f"H1-{int(start)}.hdf5")
    
    results = {}
    
    try:
        # Check Cache or Fetch L1
        if os.path.exists(cache_l1):
            print('loading cache', cache_l1)
            data_l1 = TimeSeries.read(cache_l1)
        else:
            print('downloading', cache_l1)
            data_l1 = TimeSeries.fetch_open_data('L1', start, start + duration, cache=True)
            data_l1.write(cache_l1, overwrite=True)
        results['L1'] = np.std(data_l1.detrend().value)
        
        # Check Cache or Fetch H1
        if os.path.exists(cache_h1):
            print('loading cache', cache_h1)
            data_h1 = TimeSeries.read(cache_h1)
        else:
            print('downloading', cache_h1)
            data_h1 = TimeSeries.fetch_open_data('H1', start, start + duration, cache=True)
            data_h1.write(cache_h1, overwrite=True)
        results['H1'] = np.std(data_h1.detrend().value)
        
        return results
    except Exception as err:
        # If either fetch fails, this snapshot is invalid for a strict test
        print('DOWNLOAD ERROR', err)
        return None

def main():
    print("--- STARTING STRICT DUAL-DETECTOR ANALYSIS (L1 + H1) ---")
    start_time = time.time()
    download_and_extract()
    fits_files = sorted([f for f in os.listdir(SUB_FOLDER) if f.endswith('.fits')])
    pulsar_points = []

    # 1. Map all Pulsar Snapshots
    for f in fits_files:
        try:
            with fits.open(os.path.join(SUB_FOLDER, f)) as h:
                hdr = h[0].header
                gps = (hdr['STT_IMJD'] - 44244.0) * 86400 + hdr['STT_SMJD'] - 32 + h['SUBINT'].data['OFFS_SUB'][0]
                psr_sum = np.sum(h['SUBINT'].data['DATA'])
                pulsar_points.append({'gps': gps, 'psr': psr_sum})
        except: continue

    print(f"[INFO] Found {len(pulsar_points)} snapshots. Verifying Dual-Detector Lock...")

    final_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(get_dual_ligo_data, p['gps']): p for p in pulsar_points}
        
        for i, future in enumerate(as_completed(future_map)):
            p = future_map[future]
            ligo_data = future.result()
            
            if ligo_data: # Only proceeds if BOTH L1 and H1 were found
                final_results.append({
                    'gps': p['gps'], 
                    'psr': p['psr'], 
                    'l1': ligo_data['L1'], 
                    'h1': ligo_data['H1']
                })
            
            if i % 5 == 0:
                print(f"  [CHECKING] {i}/{len(final_results)}... Valid Pairs Found: {len(final_results)}")

    if not final_results:
        print("\n[FAIL] No snapshots had simultaneous L1/H1 coverage.")
        return

    # 2. Sort and Process
    final_results.sort(key=lambda x: x['gps'])
    gps_arr = np.array([d['gps'] for d in final_results])
    psr_n = (np.array([d['psr'] for d in final_results]) - np.mean([d['psr'] for d in final_results])) / np.std([d['psr'] for d in final_results])
    l1_n = (np.array([d['l1'] for d in final_results]) - np.mean([d['l1'] for d in final_results])) / np.std([d['l1'] for d in final_results])
    h1_n = (np.array([d['h1'] for d in final_results]) - np.mean([d['h1'] for d in final_results])) / np.std([d['h1'] for d in final_results])

    # 3. Visualization
    plt.figure(figsize=(14, 8))
    
    # Pulsar vs L1
    plt.subplot(2, 1, 1)
    plt.plot(psr_n, 'o-', label="Pulsar J1713", color='cyan', markersize=4)
    plt.plot(l1_n, 's--', label="LIGO Livingston (L1)", color='red', alpha=0.5)
    r_l1 = np.corrcoef(psr_n, l1_n)[0,1]
    plt.title(f"Strict Correlation: Pulsar vs L1 (R={r_l1:.4f})")
    plt.legend()

    # Pulsar vs H1
    plt.subplot(2, 1, 2)
    plt.plot(psr_n, 'o-', label="Pulsar J1713", color='cyan', markersize=4)
    plt.plot(h1_n, 's--', label="LIGO Hanford (H1)", color='blue', alpha=0.5)
    r_h1 = np.corrcoef(psr_n, h1_n)[0,1]
    plt.title(f"Strict Correlation: Pulsar vs H1 (R={r_h1:.4f})")
    plt.legend()

    plt.tight_layout()
    
    # Global Metric: Average Correlation
    print(f"\n--- FINAL ANALYSIS ---")
    print(f"Total Valid Triplets: {len(final_results)}")
    print(f"Mean Correlation (L1/H1 Combined): {(r_l1 + r_h1)/2:.4f}")
    
    plt.show()

if __name__ == "__main__":
    main()