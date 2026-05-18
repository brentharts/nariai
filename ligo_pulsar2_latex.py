import os
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from gwpy.timeseries import TimeSeries
from gwpy.time import tconvert
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIG ---
SUB_FOLDER = "pulsar_fits_large/Rcvr1_2-VEGAS"
CACHE_DIR = "ligo_cache"
MAX_WORKERS = 4 

def get_dual_ligo_data(start, duration=30):
    """Fetches BOTH L1 and H1 data streams. Aborts early if NaNs are found."""
    cache_l1 = os.path.join(CACHE_DIR, f"L1-{int(start)}.hdf5")
    cache_h1 = os.path.join(CACHE_DIR, f"H1-{int(start)}.hdf5")
    results = {}
    try:
        # L1 Pipeline Processing
        if os.path.exists(cache_l1): 
            data_l1 = TimeSeries.read(cache_l1)
        else: 
            data_l1 = TimeSeries.fetch_open_data('L1', start, start+duration)
            data_l1.write(cache_l1, overwrite=True)
        
        # H1 Pipeline Processing
        if os.path.exists(cache_h1): 
            data_h1 = TimeSeries.read(cache_h1)
        else: 
            data_h1 = TimeSeries.fetch_open_data('H1', start, start+duration)
            data_h1.write(cache_h1, overwrite=True)
        
        l1_val = np.std(data_l1.detrend().value)
        h1_val = np.std(data_h1.detrend().value)
        
        # Eliminate corrupted data slices immediately at the source
        if np.isnan(l1_val) or np.isnan(h1_val): 
            return None
        return {'L1': l1_val, 'H1': h1_val}
    except Exception: 
        return None

def export_calibrated_sync_to_latex(final_triplets, psr_z, l1_z, h1_z, r_l1, r_h1, r_x, filename):
    """Generates an optimized standalone triple-trace PGFPlots synchronization asset."""
    print(f"Generating optimized LaTeX TikZ/PGFPlots calibrated pulsar asset: {filename}...")
    
    # --- Generate Standardized Data Table Rows ---
    table_rows = ""
    for idx in range(len(final_triplets)):
        table_rows += f"{idx} {psr_z[idx]:.4f} {l1_z[idx]:.4f} {h1_z[idx]:.4f}\n"

    latex_template = rf"""\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.90\textwidth,
    height=0.45\textwidth,
    title={{Non-Local Vacuum Synchronization (Calibrated Intensity)}},
    xlabel={{Snapshot Sequential Index}},
    ylabel={{Standardized Variation ($Z$)}},
    xmin=0,
    xmax={len(final_triplets)-1},
    grid=both,
    grid style={{dashed, gray!10}},
    legend style={{at={{(0.5,-0.2)}}, anchor=north, legend columns=3, nodes={{scale=0.8, transform shape}}}},
    every axis title/.style={{font=\bfseries\small}}
]

% 1. Calibrated Pulsar Absolute Intensity Trace
\addplot[color=yellow!90!black, thick, mark=*] table [x=idx, y=psr] {{
idx psr l1 h1
{table_rows}}};
\addlegendentry{{Pulsar Intensity}}

% 2. LIGO Livingston (L1) Variance Trace
\addplot[color=red, dashed, thick, mark=square*] table [x=idx, y=l1] {{
idx psr l1 h1
{table_rows}}};
\addlegendentry{{L1 Jitter}}

% 3. LIGO Hanford (H1) Variance Trace
\addplot[color=blue, dotted, thick, mark=triangle*] table [x=idx, y=h1] {{
idx psr l1 h1
{table_rows}}};
\addlegendentry{{H1 Jitter}}

\end{{axis}}
\end{{tikzpicture}}
\caption{{Synchronized amplitude tracking comparing calibrated absolute profile intensities from the Green Bank Telescope (PSR J1713+0747) against localized detector variances across $N={len(final_triplets)}$ concurrent data frames. Linear correlation mapping coefficients track at $R_{{\text{{L1}}}} = {r_l1:.4f}$ and $R_{{\text{{H1}}}} = {r_h1:.4f}$, with an inter-detector metric of $R_{{\text{{H1-L1}}}} = {r_x:.4f}$.}}
\label{{fig:calibrated_vacuum_sync}}
\end{{figure}}
"""

    with open(filename, "w") as f:
        f.write(latex_template)
    print(f"LaTeX file successfully saved to {filename}")

def main():
    print("\n" + "="*60)
    print("      SPECTRE VACUUM CORRELATOR: CALIBRATED")
    print("="*60)
    
    if not os.path.exists(SUB_FOLDER):
        print(f"[FAIL] Target directory '{SUB_FOLDER}' not found.")
        return
        
    fits_files = sorted([f for f in os.listdir(SUB_FOLDER) if f.endswith('.fits')])
    raw_data = []

    print(f"[STAGE 1] Extracting profiles from {len(fits_files)} PSRFITS file matrices...")
    for f in fits_files:
        try:
            with fits.open(os.path.join(SUB_FOLDER, f)) as h:
                hdr = h[0].header
                gps = (hdr['STT_IMJD'] - 44244.0) * 86400 + hdr['STT_SMJD'] - 32 + h['SUBINT'].data['OFFS_SUB'][0]
                
                # Take absolute value and mean of the pulsar matrix 
                # to correct for negative baselines in the VEGAS data.
                psr_raw = h['SUBINT'].data['DATA']
                psr_intensity = float(np.abs(np.mean(psr_raw)))
                
                raw_data.append({'gps': gps, 'psr': psr_intensity})
        except Exception: 
            continue

    print(f"[STAGE 2] Found {len(raw_data)} snaps. Syncing with LIGO...")

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
        print("[FAIL] No valid triplets survived the cross-instrument synchronization thresholds.")
        return

    final_triplets.sort(key=lambda x: x['gps'])
    
    # Extract independent data vectors
    psr = np.array([d['psr'] for d in final_triplets])
    l1  = np.array([d['l1'] for d in final_triplets])
    h1  = np.array([d['h1'] for d in final_triplets])

    # Calculate true linear Pearson Correlation Coefficients
    r_l1 = np.corrcoef(psr, l1)[0, 1]
    r_h1 = np.corrcoef(psr, h1)[0, 1]
    r_cross = np.corrcoef(l1, h1)[0, 1]

    print("\n" + "="*60)
    print(f"FINAL PHYSICS REPORT (N={len(final_triplets)})")
    print("="*60)
    print(f"Pulsar-Livingston Sync: {r_l1:.6f}")
    print(f"Pulsar-Hanford Sync:    {r_h1:.6f}")
    print(f"Cross-Detector Sync:    {r_cross:.6f}")
    print("="*60 + "\n")

    # Z-Score standardization for clean mathematical layout rendering
    psr_z = (psr - np.mean(psr)) / np.std(psr)
    l1_z = (l1 - np.mean(l1)) / np.std(l1)
    h1_z = (h1 - np.mean(h1)) / np.std(h1)

    # Export structured PGFPlots documentation file
    name = os.path.split(__file__)[-1]
    tex_filename = f"{name}_CalibratedSync.tex"
    export_calibrated_sync_to_latex(final_triplets, psr_z, l1_z, h1_z, r_l1, r_h1, r_cross, filename=tex_filename)

    # Native Verification Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(psr_z, 'o-', label="Pulsar (Abs Intensity)", color='yellow')
    plt.plot(l1_z, 'r--', label="L1 Vacuum Jitter")
    plt.plot(h1_z, 'b--', label="H1 Vacuum Jitter")
    plt.title("Non-Local Vacuum Synchronization (Calibrated Z-Scores)")
    plt.xlabel("Snapshot Sequential Index")
    plt.ylabel("Standardized Amplitudes (Z)")
    plt.legend(ncol=3)
    plt.grid(alpha=0.2)
    plt.show()

if __name__ == "__main__":
    main()
