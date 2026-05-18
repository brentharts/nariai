import os
import time
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
    """Measures normalized complexity/entropy density of the input sequence."""
    x = np.array(time_series, dtype=float)
    if len(x) < order: 
        return 0.0
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
    """Retrieves dual-detector data, applies whitening, and computes entropy."""
    PAD = 4
    try:
        data = {}
        for det in ['L1', 'H1']:
            cache_path = os.path.join(CACHE_DIR, f"{det}-{int(start)}.hdf5")
            if os.path.exists(cache_path):
                ts = TimeSeries.read(cache_path)
            else:
                # Fetch raw data backed by a defensive wing padding allocation
                ts = TimeSeries.fetch_open_data(det, start - PAD, start + duration + PAD)
                ts.write(cache_path, overwrite=True)
            
            # EXECUTION FIX: Whiten the time series before stripping the padding 
            # to insulate vacuum metrics from raw 1/f noise profiles.
            clean_ts = ts.detrend().whiten().crop(start, start + duration)
            
            if np.isnan(clean_ts.value).any():
                return None
                
            ent = float(permutation_entropy(clean_ts.value, order=3))
            
            # Avoid the 'flat-line' 0.0 entropy glitches found in previous runs
            if ent < 0.01: 
                return None 
            data[det] = ent
        return data
    except Exception: 
        return None

def export_mi_to_latex(results, mi_l1, mi_h1, mi_detectors, filename):
    """Generates an optimized standalone scatter-plot asset using PGFPlots."""
    print(f"Generating optimized LaTeX TikZ/PGFPlots MI scatter asset: {filename}...")
    
    # --- Generate Space-Separated Phase Table ---
    table_rows = ""
    for r in results:
        table_rows += f"{r['psr']:.5f} {r['l1']:.5f} {r['h1']:.5f}\n"

    # Isolate vector limits to pad coordinate bounds
    psr_arr = np.array([r['psr'] for r in results])
    vac_arr = np.array([r['l1'] for r in results] + [r['h1'] for r in results])
    
    x_min, x_max = float(np.min(psr_arr) - 0.01), float(np.max(psr_arr) + 0.01)
    y_min, y_max = float(np.min(vac_arr) - 0.01), float(np.max(vac_arr) + 0.01)

    latex_template = rf"""\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.88\textwidth,
    height=0.45\textwidth,
    title={{Phase-Space Mapping of Non-Local Sync}},
    xlabel={{Pulsar Entropy (Source)}},
    ylabel={{Vacuum Entropy (Detector)}},
    xmin={x_min:.3f},
    xmax={x_max:.3f},
    ymin={y_min:.3f},
    ymax={y_max:.3f},
    grid=both,
    grid style={{dashed, gray!10}},
    legend style={{at={{(0.5,-0.2)}}, anchor=north, legend columns=2, nodes={{scale=0.8, transform shape}}}},
    every axis title/.style={{font=\bfseries\small}}
]

% 1. L1 Scatter Marks
\addplot[color=red, only marks, mark=*, mark size=2pt, fill opacity=0.6] table [x=psr, y=l1] {{
psr l1 h1
{table_rows}}};
\addlegendentry{{L1 Vacuum Space ($I = {mi_l1:.4f}$ bits)}}

% 2. H1 Scatter Marks
\addplot[color=blue, only marks, mark=square*, mark size=2pt, fill opacity=0.6] table [x=psr, y=h1] {{
psr l1 h1
{table_rows}}};
\addlegendentry{{H1 Vacuum Space ($I = {mi_h1:.4f}$ bits)}}

\end{{axis}}
\end{{tikzpicture}}
\caption{{Phase-space alignment comparing non-linear entropic variables extracted from GBT pulsar monitoring streams against whitened LIGO Hanford and Livingston channels ($N={len(results)}$ pairs). Extracted mutual information properties measure non-linear couplings at $I_{{\text{{Pulsar-L1}}}} = {mi_l1:.4f}$~bits and $I_{{\text{{Pulsar-H1}}}} = {mi_h1:.4f}$~bits, while the cross-interferometer baseline maps a mutual info capacity of $I_{{\text{{L1-H1}}}} = {mi_detectors:.4f}$~bits.}}
\label{{fig:mutual_info_phase_space}}
\end{{figure}}
"""

    with open(filename, "w") as f:
        f.write(latex_template)
    print(f"LaTeX file successfully saved to {filename}")

def main():
    print("\n" + "="*70)
    print("      NON-LOCAL VACUUM SYNC: MUTUAL INFORMATION (MI) ANALYSIS")
    print("="*70)
    
    if not os.path.exists(SUB_FOLDER):
        print(f"[FAIL] Target archive path '{SUB_FOLDER}' not found.")
        return
        
    fits_files = sorted([f for f in os.listdir(SUB_FOLDER) if f.endswith('.fits')])
    raw_points = []

    print(f"[STAGE 1] Extracting informational distributions from {len(fits_files)} FITS structures...")
    for f in fits_files:
        try:
            with fits.open(os.path.join(SUB_FOLDER, f)) as h:
                hdr = h[0].header
                gps = (hdr['STT_IMJD'] - 44244.0) * 86400 + hdr['STT_SMJD'] - 32 + h['SUBINT'].data['OFFS_SUB'][0]
                
                # Reconstruct native 2D timeline prior to vector pooling
                psr_raw = h['SUBINT'].data['DATA']
                psr_flat = np.abs(psr_raw.reshape(-1))
                
                psr_ent = float(permutation_entropy(psr_flat, order=3))
                if psr_ent > 0.01:
                    raw_points.append({'gps': gps, 'psr_s': psr_ent})
        except Exception: 
            continue

    print(f"[STAGE 2] Running multi-threaded informational cross-match query loops...")
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
        print(f"[!] Not enough valid triplets survived tracking thresholds ({len(results)} found). Minimum required is 5.")
        return

    # Establish global time-ordering compliance
    results.sort(key=lambda x: x['gps'])
    
    psr = np.array([r['psr'] for r in results]).reshape(-1, 1)
    l1 = np.array([r['l1'] for r in results])
    h1 = np.array([r['h1'] for r in results])

    # MI Calculation: Non-parametric quantification of "Shared Information" in Bits
    mi_l1 = float(mutual_info_regression(psr, l1, random_state=42)[0])
    mi_h1 = float(mutual_info_regression(psr, h1, random_state=42)[0])
    mi_detectors = float(mutual_info_regression(l1.reshape(-1, 1), h1, random_state=42)[0])

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
    print("="*70 + "\n")

    # Export standalone PGFPlots scatter phase layout
    name = os.path.split(__file__)[-1]
    tex_filename = f"{name}_MutualInfo.tex"
    export_mi_to_latex(results, mi_l1, mi_h1, mi_detectors, filename=tex_filename)

    # Native Verification Plotting
    plt.figure(figsize=(10, 5))
    plt.scatter(psr.flatten(), l1, color='red', alpha=0.6, edgecolors='none', label=f'L1 Entropy (MI={mi_l1:.3f})')
    plt.scatter(psr.flatten(), h1, color='blue', alpha=0.6, edgecolors='none', label=f'H1 Entropy (MI={mi_h1:.3f})')
    plt.xlabel("Pulsar Entropy (Source)")
    plt.ylabel("Vacuum Entropy (Detector)")
    plt.title("Phase-Space Mapping of Non-Local Sync")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
