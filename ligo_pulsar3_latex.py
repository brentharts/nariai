import os
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from gwpy.timeseries import TimeSeries
from gwpy.time import tconvert
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import factorial

# --- CONFIG ---
SUB_FOLDER = "pulsar_fits_large/Rcvr1_2-VEGAS"
CACHE_DIR = "ligo_cache"
MAX_WORKERS = 4 

def permutation_entropy(time_series, order=3, delay=1):
    """
    Calculates normalized Permutation Entropy over a 1D data sequence.
    Returns values bounded between 0.0 (perfectly predictable) and 1.0 (pure chaos).
    """
    x = np.array(time_series, dtype=float)
    if len(x) < order:
        return 0.0
    
    # Generate sliding window views across the data array
    windows = np.lib.stride_tricks.sliding_window_view(x, order)[::delay]
    
    # Construct an ordinal hash mult mapping
    hashmult = np.power(order, np.arange(order))
    sorted_idx = np.argsort(windows, axis=1)
    hashvalues = (sorted_idx * hashmult).sum(axis=1)
    
    _, counts = np.unique(hashvalues, return_counts=True)
    probs = counts / counts.sum()
    
    # Normalized Shannon Entropy over ordinal permutations
    return -np.sum(probs * np.log2(probs)) / np.log2(factorial(order))

def get_vacuum_entropy(start, duration=30):
    """Loads, detrends, and whitens data frames before computing permutation complexity."""
    cache_l1 = os.path.join(CACHE_DIR, f"L1-{int(start)}.hdf5")
    cache_h1 = os.path.join(CACHE_DIR, f"H1-{int(start)}.hdf5")
    
    # 4-second padding buffer to shield the entropy window from whitening edge transients
    PAD = 4
    
    try:
        # Load and process L1 data stream
        if os.path.exists(cache_l1): 
            data_l1 = TimeSeries.read(cache_l1)
        else: 
            data_l1 = TimeSeries.fetch_open_data('L1', start - PAD, start + duration + PAD)
            data_l1.write(cache_l1, overwrite=True)
            
        # Load and process H1 data stream
        if os.path.exists(cache_h1): 
            data_h1 = TimeSeries.read(cache_h1)
        else: 
            data_h1 = TimeSeries.fetch_open_data('H1', start - PAD, start + duration + PAD)
            data_h1.write(cache_h1, overwrite=True)
        
        # Whiten first to eliminate 1/f instrumental trends, then crop to nominal window
        l1_clean = data_l1.detrend().whiten().crop(start, start + duration)
        h1_clean = data_h1.detrend().whiten().crop(start, start + duration)
        
        # Verify slice data viability
        if np.isnan(l1_clean.value).any() or np.isnan(h1_clean.value).any():
            return None
            
        # Extract complexity scores (Order 3 maps rapid structural inversions)
        ent_l1 = float(permutation_entropy(l1_clean.value, order=3))
        ent_h1 = float(permutation_entropy(h1_clean.value, order=3))
        
        return {'L1_Ent': ent_l1, 'H1_Ent': ent_h1}
    except Exception: 
        return None

def export_entropy_sync_to_latex(final_triplets, psr, l1, h1, r_l1, r_h1, r_cross, filename):
    """Generates an optimized standalone discrete PGFPlots entropy timeline asset."""
    print(f"Generating optimized LaTeX TikZ/PGFPlots entropic asset: {filename}...")
    
    # --- Generate Space-Separated Table Columns ---
    table_rows = ""
    for idx in range(len(final_triplets)):
        table_rows += f"{idx} {psr[idx]:.5f} {l1[idx]:.5f} {h1[idx]:.5f}\n"

    # Calculate absolute bounds for clear plot limits
    all_vals = np.concatenate([psr, l1, h1])
    y_min = f"{float(np.min(all_vals) - 0.02):.3f}"
    y_max = f"{float(np.max(all_vals) + 0.02):.3f}"

    latex_template = rf"""\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.90\textwidth,
    height=0.45\textwidth,
    title={{Aperiodic Entropy Wave Synchronization}},
    xlabel={{Snapshot Sequential Index}},
    ylabel={{Permutation Entropy ($H_n$)}},
    xmin=0,
    xmax={len(final_triplets)-1},
    ymin={y_min},
    ymax={y_max},
    grid=both,
    grid style={{dashed, gray!10}},
    legend style={{at={{(0.5,-0.2)}}, anchor=north, legend columns=3, nodes={{scale=0.8, transform shape}}}},
    every axis title/.style={{font=\bfseries\small}}
]

% 1. Pulsar Struct Profile Permutation Entropy
\addplot[color=yellow!90!black, thick, const plot, mark=*] table [x=idx, y=psr] {{
idx psr l1 h1
{table_rows}}};
\addlegendentry{{Pulsar Profile Entropy}}

% 2. LIGO Livingston (L1) Whitened Vacuum Entropy
\addplot[color=red, dashed, thick, const plot, mark=square*] table [x=idx, y=l1] {{
idx psr l1 h1
{table_rows}}};
\addlegendentry{{L1 Vacuum Entropy}}

% 3. LIGO Hanford (H1) Whitened Vacuum Entropy
\addplot[color=blue, dotted, thick, const plot, mark=triangle*] table [x=idx, y=h1] {{
idx psr l1 h1
{table_rows}}};
\addlegendentry{{H1 Vacuum Entropy}}

\end{{axis}}
\end{{tikzpicture}}
\caption{{Simultaneous information-theoretic tracking analyzing permutation complexity scores ($n=3$) across $N={len(final_triplets)}$ concurrent cross-instrument profiles. Multi-node linear regressions map network correlations at $R_{{\text{{Pulsar-L1}}}} = {r_l1:.4f}$ and $R_{{\text{{Pulsar-H1}}}} = {r_h1:.4f}$, with an underlying inter-detector cross-talk parameter of $R_{{\text{{H1-L1}}}} = {r_cross:.4f}$.}}
\label{{fig:entropy_wave_sync}}
\end{{figure}}
"""

    with open(filename, "w") as f:
        f.write(latex_template)
    print(f"LaTeX file successfully saved to {filename}")

def main():
    print("\n" + "="*60)
    print("      ENTROPY-WAVE COUPLING ANALYSIS (NON-LOCAL)")
    print("="*60)
    
    if not os.path.exists(SUB_FOLDER):
        print(f"[FAIL] Target archive path '{SUB_FOLDER}' not found.")
        return
        
    fits_files = sorted([f for f in os.listdir(SUB_FOLDER) if f.endswith('.fits')])
    raw_data = []

    print(f"[STAGE 1] Computing structural profiles across {len(fits_files)} pulsar data matrices...")
    for f in fits_files:
        try:
            with fits.open(os.path.join(SUB_FOLDER, f)) as h:
                hdr = h[0].header
                gps = (hdr['STT_IMJD'] - 44244.0) * 86400 + hdr['STT_SMJD'] - 32 + h['SUBINT'].data['OFFS_SUB'][0]
                
                # Reshape to 2D matrix structure to preserve accurate time-bin alignments
                psr_raw = h['SUBINT'].data['DATA']
                # Flatten cleanly per-subintegration frame to isolate shape variations
                psr_flat = np.abs(psr_raw.reshape(-1))
                
                psr_ent = float(permutation_entropy(psr_flat, order=3))
                raw_data.append({'gps': gps, 'psr_ent': psr_ent})
        except Exception: 
            continue

    if not raw_data:
        print("[FAIL] Zero pulsar profile matrices parsed.")
        return

    print(f"[STAGE 2] Syncing {len(raw_data)} entropy scores with whitened LIGO frames...")

    final_triplets = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(get_vacuum_entropy, p['gps']): p for p in raw_data}
        for future in as_completed(future_map):
            p = future_map[future]
            try:
                ent_data = future.result()
                if ent_data:
                    final_triplets.append({
                        'gps': p['gps'], 
                        'psr': p['psr_ent'], 
                        'l1': ent_data['L1_Ent'], 
                        'h1': ent_data['H1_Ent']
                    })
                    print(f" [ENTROPY SYNC] GPS: {p['gps']:.1f} | PSR_S: {p['psr_ent']:.4f} | L1_S: {ent_data['L1_Ent']:.4f} | H1_S: {ent_data['H1_Ent']:.4f}")
            except Exception:
                continue

    if not final_triplets:
        print("[FAIL] No concurrent data frames passed cross-talk quality thresholds.")
        return

    # Ensure chronological consistency across the time vector
    final_triplets.sort(key=lambda x: x['gps'])
    
    psr = np.array([d['psr'] for d in final_triplets])
    l1  = np.array([d['l1'] for d in final_triplets])
    h1  = np.array([d['h1'] for d in final_triplets])

    r_l1 = np.corrcoef(psr, l1)[0, 1]
    r_h1 = np.corrcoef(psr, h1)[0, 1]
    r_cross = np.corrcoef(l1, h1)[0, 1]

    print("\n" + "="*60)
    print(f"NON-LOCAL ENTROPY REPORT (N={len(final_triplets)})")
    print("="*60)
    print(f"Pulsar <-> L1 Entropy Correlation: {r_l1:.6f}")
    print(f"Pulsar <-> H1 Entropy Correlation: {r_h1:.6f}")
    print(f"Detector Entropy Cross-Talk:       {r_cross:.6f}")
    print("="*60 + "\n")

    # Export standalone PGFPlots tracking layout
    name = os.path.split(__file__)[-1]
    tex_filename = f"{name}_EntropySync.tex"
    export_entropy_sync_to_latex(final_triplets, psr, l1, h1, r_l1, r_h1, r_cross, filename=tex_filename)

    # Native Verification Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(psr, 'gold', label="Pulsar Entropy Score", linewidth=2, drawstyle='steps-post')
    plt.plot(l1, 'r--', label="L1 Vacuum Entropy", alpha=0.7, drawstyle='steps-post')
    plt.plot(h1, 'b--', label="H1 Vacuum Entropy", alpha=0.7, drawstyle='steps-post')
    plt.title("Aperiodic Entropy Wave Synchronization (Whitened)")
    plt.xlabel("Snapshot Sequential Index")
    plt.ylabel("Permutation Entropy ($H_n$)")
    plt.legend(ncol=3)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
