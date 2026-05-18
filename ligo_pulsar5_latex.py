import os
import time
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
ROLLING_WINDOW = 8  # Number of snapshots to calculate shared information over

def permutation_entropy(time_series, order=3, delay=1):
    """Calculates Permutation Entropy (Proxy for Vacuum Complexity)."""
    x = np.array(time_series, dtype=float)
    if len(x) < order: 
        return 0.0
    sw = np.lib.stride_tricks.sliding_window_view(x, order)[::delay]
    sorted_idx = np.argsort(sw, axis=1)
    hashmult = np.power(order, np.arange(order))
    hashvalues = (sorted_idx * hashmult).sum(1)
    _, counts = np.unique(hashvalues, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs)) / np.log2(factorial(order))

def get_vacuum_entropy_pair(start, duration=30):
    """Fetches L1 and H1 data, applies whitening filters, and returns entropy scores."""
    PAD = 4
    try:
        data = {}
        for det in ['L1', 'H1']:
            cache_path = os.path.join(CACHE_DIR, f"{det}-{int(start)}.hdf5")
            if os.path.exists(cache_path):
                ts = TimeSeries.read(cache_path)
            else:
                # Fetch raw data backed by a defensive padding allocation
                ts = TimeSeries.fetch_open_data(det, start - PAD, start + duration + PAD)
                ts.write(cache_path, overwrite=True)
            
            # EXECUTION FIX: Whiten the time series before stripping the padding
            clean_ts = ts.detrend().whiten().crop(start, start + duration)
            
            if np.isnan(clean_ts.value).any():
                return None
                
            ent = float(permutation_entropy(clean_ts.value, order=3))
            
            # Filter out flat-line/glitch segments
            if ent < 0.05: 
                return None 
            data[det] = ent
        return data
    except Exception:
        return None

def export_rolling_bridge_to_latex(timeline, energy_flux, bridge_mi, coupling, filename):
    """Generates an optimized standalone dual-axis PGFPlots compilation tracking metric sync."""
    print(f"Generating optimized LaTeX TikZ/PGFPlots rolling dual-axis asset: {filename}...")
    
    # --- Generate Space-Separated Unified Data Columns ---
    table_rows = ""
    for i in range(len(timeline)):
        table_rows += f"{timeline[i]:.1f} {energy_flux[i]:.5e} {bridge_mi[i]:.5f}\n"

    x_min, x_max = timeline[0], timeline[-1]
    
    # Format limits elegantly
    flux_min, flux_max = np.min(energy_flux) * 0.95, np.max(energy_flux) * 1.05
    mi_min, mi_max = np.min(bridge_mi) - 0.05, np.max(bridge_mi) + 0.05

    latex_template = rf"""\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.90\textwidth,
    height=0.45\textwidth,
    title={{The Spectre Mechanism: Rolling Information Bridge Bounds}},
    xlabel={{GPS Timestamp (Seconds)}},
    xmin={x_min:.1f},
    xmax={x_max:.1f},
    ymin={flux_min:.5e},
    ymax={flux_max:.5e},
    axis y line*=left,
    ylabel={{\color{{yellow!80!black}}Pulsar Flux (Energy Density)}},
    ylabel style={{color={{yellow!80!black}}}},
    tick label style={{/pgf/number format/fixed}},
    grid=both,
    grid style={{dashed, gray!10}},
    every axis title/.style={{font=\bfseries\small}}
]

% 1. Left Axis - Pulsar Energy Density Profile
\addplot[color=yellow!80!black, ultra thick] table [x=gps, y=flux] {{
gps flux mi
{table_rows}}};
\label{{plot:flux}}
\end{{axis}}

\begin{{axis}}[
    width=0.90\textwidth,
    height=0.45\textwidth,
    xmin={x_min:.1f},
    xmax={x_max:.1f},
    ymin={mi_min:.3f},
    ymax={mi_max:.3f},
    axis x line=none,
    axis y line*=right,
    ylabel={{\color{{cyan!80!black}}L1-H1 Shared Info Bridge (Bits)}},
    ylabel style={{color={{cyan!80!black}}}},
    legend style={{at={{(0.5,-0.2)}}, anchor=north, legend columns=2, nodes={{scale=0.8, transform shape}}}}
]
\addlegendimage{{/pgfplots/refstyle=plot:flux}}\addlegendentry{{Pulsar Energy Flux}}

% 2. Right Axis - Shared Information Bridge (MI)
\addplot[color=cyan!80!black, dashed, thick, mark=*] table [x=gps, y=mi] {{
gps flux mi
{table_rows}}};
\addlegendentry{{Information Bridge ($I_{{\text{{L1-H1}}}}$)}}

\end{{axis}}
\end{{tikzpicture}}
\caption{{Dynamic evaluation mapping the tracking profile across $N={len(timeline)}$ chronological step metrics (Rolling Window Size $= {ROLLING_WINDOW}$). The cross-correlation metric tracking changes in local pulsar energy density against the non-linear vacuum informational bridge evaluates to $R_{{\text{{Coupling}}}} = {coupling:.6f}$.}}
\label{{fig:spectre_rolling_bridge}}
\end{{figure}}
"""

    with open(filename, "w") as f:
        f.write(latex_template)
    print(f"LaTeX file successfully saved to {filename}")

def main():
    print("\n" + "="*80)
    print("      SPECTRE BRIDGE ANALYSIS: UNIFIED NON-LOCAL ENTROPY FLOW")
    print("="*80)
    
    if not os.path.exists(CACHE_DIR): 
        os.makedirs(CACHE_DIR)
    
    if not os.path.exists(SUB_FOLDER):
        print(f"[FAIL] Target archive path '{SUB_FOLDER}' not found.")
        return
        
    fits_files = sorted([f for f in os.listdir(SUB_FOLDER) if f.endswith('.fits')])
    print(f"[DEBUG] Found {len(fits_files)} pulsar FITS files. Harvesting timestamps...")

    raw_list = []
    for f in fits_files:
        try:
            with fits.open(os.path.join(SUB_FOLDER, f)) as h:
                hdr = h[0].header
                gps = (hdr['STT_IMJD'] - 44244.0) * 86400 + hdr['STT_SMJD'] - 32 + h['SUBINT'].data['OFFS_SUB'][0]
                
                # Structural Shape Tracking Verification
                psr_raw = h['SUBINT'].data['DATA']
                psr_flat = np.abs(psr_raw.reshape(-1))
                
                psr_ent = permutation_entropy(psr_flat, order=3)
                psr_intensity = np.mean(psr_flat)
                
                if psr_ent > 0.05:
                    raw_list.append({'gps': gps, 'psr_ent': psr_ent, 'psr_flux': psr_intensity})
        except Exception: 
            continue

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
        print(f"[FAIL] Insufficient synced data points ({len(final_data)} found) for rolling analysis window size of {ROLLING_WINDOW}.")
        return

    # Sort chronologically to preserve accurate time-series step dependencies
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
        
        # Cross-Detector Bridge Strength via Non-Parametric Mutual Information
        mi_score = float(mutual_info_regression(win_l1, win_h1, random_state=42)[0])
        
        bridge_mi.append(mi_score)
        energy_flux.append(float(np.mean(win_psr)))
        timeline.append(float(gps_arr[i + ROLLING_WINDOW // 2]))

    # Calculate final coupling factor
    coupling = float(np.corrcoef(energy_flux, bridge_mi)[0, 1])

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
    print("="*80 + "\n")

    # Export standalone PGFPlots layout asset
    name = os.path.split(__file__)[-1]
    tex_filename = f"{name}_RollingBridge.tex"
    export_rolling_bridge_to_latex(timeline, energy_flux, bridge_mi, coupling, filename=tex_filename)

    # Native Verification Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('GPS Time (Seconds)')
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
