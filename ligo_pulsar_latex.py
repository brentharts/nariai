import os
import tarfile
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from gwpy.timeseries import TimeSeries
from gwpy.time import tconvert
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIG ---
CACHE_DIR = "ligo_cache"
MAX_WORKERS = 2  # Balanced for parallel dual-stream downloads
ZENODO_URL = 'https://zenodo.org/records/10622185/files/Rcvr1_2-VEGAS-fits.tar.gz'
LOCAL_TAR = "psr_large_data.tar.gz"
EXTRACT_DIR = "pulsar_fits_large"
SUB_FOLDER = os.path.join(EXTRACT_DIR, "Rcvr1_2-VEGAS")

def download_and_extract():
    """Downloads large PSRFITS archive cleanly using a streaming connection manager."""
    if not os.path.exists(LOCAL_TAR):
        print(f"[1/3] Downloading Large PSRFITS archive (770MB). This will take a few minutes...")
        with requests.get(ZENODO_URL, stream=True) as r:
            r.raise_for_status()
            with open(LOCAL_TAR, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    
    if not os.path.exists(EXTRACT_DIR):
        print(f"[2/3] Extracting archive...")
        with tarfile.open(LOCAL_TAR, "r:gz") as tar:
            tar.extractall(path=EXTRACT_DIR)

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_dual_ligo_data(start, duration=30):
    """Fetches BOTH L1 and H1. Fails if either is unavailable or contains NaNs."""
    cache_l1 = os.path.join(CACHE_DIR, f"L1-{int(start)}.hdf5")
    cache_h1 = os.path.join(CACHE_DIR, f"H1-{int(start)}.hdf5")
    results = {}
    
    try:
        # Check Cache or Fetch L1
        if os.path.exists(cache_l1):
            data_l1 = TimeSeries.read(cache_l1)
        else:
            data_l1 = TimeSeries.fetch_open_data('L1', start, start + duration, cache=True)
            data_l1.write(cache_l1, overwrite=True)
            
        # Check Cache or Fetch H1
        if os.path.exists(cache_h1):
            data_h1 = TimeSeries.read(cache_h1)
        else:
            data_h1 = TimeSeries.fetch_open_data('H1', start, start + duration, cache=True)
            data_h1.write(cache_h1, overwrite=True)
            
        # Drop frame if data contains dropouts or instrument NaNs
        if np.isnan(data_l1.value).any() or np.isnan(data_h1.value).any():
            return None
            
        results['L1'] = float(np.std(data_l1.detrend().value))
        results['H1'] = float(np.std(data_h1.detrend().value))
        return results
        
    except Exception as err:
        print(f"  [DOWNLOAD ERROR] GPS {start}: {err}")
        return None

def export_pulsar_to_latex(final_results, psr_n, l1_n, h1_n, r_l1, r_h1, filename):
    """Generates an optimized dual-axis PGFPlots visualization for paper compilation."""
    print(f"Generating optimized LaTeX TikZ/PGFPlots pulsar asset: {filename}...")
    
    # --- Generate Unified Data Table Rows ---
    table_rows = ""
    for idx in range(len(final_results)):
        table_rows += f"{idx} {psr_n[idx]:.4f} {l1_n[idx]:.4f} {h1_n[idx]:.4f}\n"

    mean_global_r = (r_l1 + r_h1) / 2.0

    latex_template = rf"""\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.85\textwidth,
    height=0.38\textwidth,
    title={{Strict Path Identity Correlation: Pulsar vs. L1}},
    xlabel={{Snapshot Index}},
    ylabel={{Normalized Variance ($\sigma$)}},
    xmin=0,
    xmax={len(final_results)-1},
    grid=both,
    grid style={{dashed, gray!10}},
    legend pos=north east,
    legend style={{nodes={{scale=0.7, transform shape}}}},
    every axis title/.style={{font=\bfseries\small}}
]
\addplot[color=cyan, thick, mark=*] table [x=idx, y=psr] {{
idx psr l1 h1
{table_rows}}};
\addlegendentry{{Pulsar J1713 ($R = {r_l1:.4f}$)}}

\addplot[color=red, dashed, thick, mark=square*] table [x=idx, y=l1] {{
idx psr l1 h1
{table_rows}}};
\addlegendentry{{LIGO L1}}
\end{{axis}}
\end{{tikzpicture}}

\vspace{{0.5cm}}

\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.85\textwidth,
    height=0.38\textwidth,
    title={{Strict Path Identity Correlation: Pulsar vs. H1}},
    xlabel={{Snapshot Index}},
    ylabel={{Normalized Variance ($\sigma$)}},
    xmin=0,
    xmax={len(final_results)-1},
    grid=both,
    grid style={{dashed, gray!10}},
    legend pos=north east,
    legend style={{nodes={{scale=0.7, transform shape}}}},
    every axis title/.style={{font=\bfseries\small}}
]
\addplot[color=cyan, thick, mark=*] table [x=idx, y=psr] {{
idx psr l1 h1
{table_rows}}};
\addlegendentry{{Pulsar J1713 ($R = {r_h1:.4f}$)}}

\addplot[color=blue, dashed, thick, mark=square*] table [x=idx, y=h1] {{
idx psr l1 h1
{table_rows}}};
\addlegendentry{{LIGO H1}}
\end{{axis}}
\end{{tikzpicture}}
\caption{{Simultaneous tracking comparing Normalized Green Bank Telescope PSR J1713+0747 profile integration values against local LIGO Hanford (H1) and Livingston (L1) data frames across {len(final_results)} valid multi-instrument pairs. Global cross-network validation tracking yields a combined mean network correlation parameter of $R = {mean_global_r:.4f}$.}}
\label{{fig:pulsar_gw_correlation}}
\end{{figure}}
"""

    with open(filename, "w") as f:
        f.write(latex_template)
    print(f"LaTeX file successfully saved to {filename}")

def main():
    print("--- STARTING STRICT DUAL-DETECTOR ANALYSIS (L1 + H1) ---")
    start_time = time.time()
    
    download_and_extract()
    
    if not os.path.exists(SUB_FOLDER):
        print(f"CRITICAL ERROR: Extraction path {SUB_FOLDER} not found.")
        return
        
    fits_files = sorted([f for f in os.listdir(SUB_FOLDER) if f.endswith('.fits')])
    pulsar_points = []

    # 1. Map all Pulsar Snapshots
    print(f"[INFO] Reading {len(fits_files)} PSRFITS file structures...")
    for f in fits_files:
        try:
            with fits.open(os.path.join(SUB_FOLDER, f)) as h:
                hdr = h[0].header
                # Reconstruct accurate absolute GPS timestamp mapping
                gps = (hdr['STT_IMJD'] - 44244.0) * 86400 + hdr['STT_SMJD'] - 32 + h['SUBINT'].data['OFFS_SUB'][0]
                psr_sum = float(np.sum(h['SUBINT'].data['DATA']))
                pulsar_points.append({'gps': gps, 'psr': psr_sum})
        except Exception: 
            continue

    if not pulsar_points:
        print("CRITICAL RUN FAILURE: No valid pulsar profile matrices recovered from FITS data.")
        return

    print(f"[INFO] Found {len(pulsar_points)} snapshots. Verifying Dual-Detector Lock via multi-threading...")

    # 2. Concurrent Network Cross-Fetch Engine
    final_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(get_dual_ligo_data, p['gps']): p for p in pulsar_points}
        
        for i, future in enumerate(as_completed(future_map)):
            p = future_map[future]
            try:
                ligo_data = future.result()
                if ligo_data:  # Only records if BOTH L1 and H1 were successfully parsed
                    final_results.append({
                        'gps': p['gps'], 
                        'psr': p['psr'], 
                        'l1': ligo_data['L1'], 
                        'h1': ligo_data['H1']
                    })
            except Exception as e:
                print(f"Execution handling failure on future mapping: {e}")
            
            if i % 5 == 0:
                print(f"  [CHECKING] File {i}/{len(pulsar_points)}... Coincident Triplets Found: {len(final_results)}")

    if not final_results:
        print("\n[FAIL] No pulsar snapshots shared co-linear timeline coverage with active L1/H1 frames.")
        return

    # Sort results sequentially along the temporal vector
    final_results.sort(key=lambda x: x['gps'])
    
    # 3. Mathematical Extraction and Normalization
    psr_vals = np.array([d['psr'] for d in final_results])
    l1_vals = np.array([d['l1'] for d in final_results])
    h1_vals = np.array([d['h1'] for d in final_results])

    psr_n = (psr_vals - np.mean(psr_vals)) / np.std(psr_vals)
    l1_n = (l1_vals - np.mean(l1_vals)) / np.std(l1_vals)
    h1_n = (h1_vals - np.mean(h1_vals)) / np.std(h1_vals)

    # Compute explicit Pearson Correlation Coefficients
    r_l1 = np.corrcoef(psr_n, l1_n)[0, 1]
    r_h1 = np.corrcoef(psr_n, h1_n)[0, 1]

    print("\n" + "="*40)
    print("FINAL ANALYSIS SUMMARY")
    print("="*40)
    print(f"Total Valid Triplets:  {len(final_results)}")
    print(f"L1 Correlation (R):    {r_l1:.4f}")
    print(f"H1 Correlation (R):    {r_h1:.4f}")
    print(f"Mean Combined System R: {(r_l1 + r_h1)/2:.4f}")
    print(f"Total Execution Time:  {time.time() - start_time:.1f}s")
    print("="*40 + "\n")

    # Export structured PGFPlots documentation file
    name = os.path.split(__file__)[-1]
    tex_filename = f"{name}_PulsarCrossCorr.tex"
    export_pulsar_to_latex(final_results, psr_n, l1_n, h1_n, r_l1, r_h1, filename=tex_filename)

    # Native Verification Plotting
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(psr_n, 'o-', label=f"Pulsar J1713", color='cyan', markersize=4)
    plt.plot(l1_n, 's--', label=f"LIGO Livingston (L1)", color='red', alpha=0.5)
    plt.title(f"Strict Correlation: Pulsar vs L1 (R={r_l1:.4f})")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.subplot(2, 1, 2)
    plt.plot(psr_n, 'o-', label=f"Pulsar J1713", color='cyan', markersize=4)
    plt.plot(h1_n, 's--', label=f"LIGO Hanford (H1)", color='blue', alpha=0.5)
    plt.title(f"Strict Correlation: Pulsar vs H1 (R={r_h1:.4f})")
    plt.ylabel("Normalized Amplitude")
    plt.xlabel("Snapshot Sequential Index")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
