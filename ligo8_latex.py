import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.time import tconvert
from scipy.signal import butter, filtfilt

def get_ligo_data(detector, start, end, cache_dir="ligo_cache"):
    if not os.path.exists(cache_dir): os.makedirs(cache_dir)
    cache_file = os.path.join(cache_dir, f"{detector}_{start}_{end}.h5")
    if os.path.exists(cache_file):
        return TimeSeries.read(cache_file)
    print(f"Cache miss. Downloading {detector} ({end-start}s)...")
    data = TimeSeries.fetch_open_data(detector, start, end, sample_rate=4096)
    data.write(cache_file, overwrite=True)
    return data

def manual_distinction_filter(data, low=20, high=1200, fs=4096):
    """Safely whitens and filters an isolated, padded time slice."""
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    white = data.whiten()
    filtered_data = filtfilt(b, a, white.value)
    return TimeSeries(filtered_data, sample_rate=fs, t0=data.t0)

def export_wavefront_to_latex(freqs, magnitudes, top_peaks, f_target, start_gps, offset, filename):
    """Generates an optimized PGFPlots representation of high-res CSD fine structures."""
    print(f"Generating optimized LaTeX TikZ/PGFPlots wavefront asset: {filename}...")
    
    t_slice = start_gps + offset
    utc_slice = tconvert(t_slice).strftime('%Y-%m-%d %H:%M:%S UTC')

    # --- Downsample Data Stream to Protect TeX Capacity Memory ---
    # Focus full resolution within ±100 Hz of the target; downsample elsewhere
    mask_fine = (freqs >= f_target - 100) & (freqs <= f_target + 100)
    
    table_rows = ""
    for idx, (f, m) in enumerate(zip(freqs, magnitudes)):
        # Keep every point inside the window; take every 4th point outside
        if mask_fine[idx] or (idx % 4 == 0):
            table_rows += f"{f:.3f} {m:.4e}\n"

    # --- Format Top Discovered Sub-Peaks as Marked Points ---
    peak_rows = ""
    for i, (f_p, m_p) in enumerate(top_peaks):
        peak_rows += f"{f_p:.3f} {m_p:.4e}\n"

    f_target_f = f"{float(f_target):.2f}"
    f_min = f"{float(f_target - 100):.2f}"
    f_max = f"{float(f_target + 100):.2f}"

    latex_template = rf"""\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.85\textwidth,
    height=0.45\textwidth,
    title={{Wavefront Reconstruction: Sub-structure near {f_target:.1f}~Hz}},
    xlabel={{Frequency (Hz)}},
    ylabel={{Correlation Magnitude}},
    xmin={f_min},
    xmax={f_max},
    ymin=0,
    grid=both,
    grid style={{dashed, gray!10}},
    legend pos=north east,
    legend style={{nodes={{scale=0.8, transform shape}}}},
    every axis title/.style={{font=\bfseries\small}}
]

% 1. Continuous high-res CSD fine-structure line
\addplot[color=blue, thick, no markers] table [x=freq, y=mag] {{
freq mag
{table_rows}}};
\addlegendentry{{CSD Fine Structure}}

% 2. Standalone marked points highlighting verified mathematical sub-peaks
\addplot[
    only marks,
    mark=triangle*,
    mark size=3.0pt,
    color=orange,
    draw=black
] table [x=freq, y=mag] {{
freq mag
{peak_rows}}};
\addlegendentry{{Identified Sub-Peaks}}

% 3. Vertical alignment marker referencing the target center
\draw[color=red, dashed, thick] (axis cs:{f_target_f}, \pgfkeysvalueof{{/pgfplots/ymin}}) -- (axis cs:{f_target_f}, \pgfkeysvalueof{{/pgfplots/ymax}});
\addlegendimage{{line legend, color=red, dashed, thick}}
\addlegendentry{{Nariai Center}}

\end{{axis}}
\end{{tikzpicture}}
\caption{{High-resolution spatial wavefront sub-structure tracking across a 32-second evaluation slice at T+{offset}s [GPS {t_slice}] ({utc_slice}). Triangles mark local micro-peaks identified immediately relative to the structural target tracking limit fixed at {f_target_f}~Hz.}}
\label{{fig:wavefront_scan_{t_slice}}}
\end{{figure}}
"""

    with open(filename, "w") as f:
        f.write(latex_template)
    print(f"LaTeX file successfully saved to {filename}")

def spatial_wavefront_scan(start_gps, target_slice_offset, r_target=1.3e6):
    c = 299792458
    f_target = c / r_target
    fs = 4096
    
    # Defensive buffer window to absorb whitening transient artifacts
    SLICE_PAD = 4
    
    t_start = start_gps + target_slice_offset
    t_end = t_start + 32
    
    print(f"\n--- PERFORMING WAVEFRONT RECONSTRUCTION (T+{target_slice_offset}s) ---")
    
    # Fetch full data framework with padded boundary limits
    h1_full = get_ligo_data('H1', t_start - SLICE_PAD, t_end + SLICE_PAD)
    l1_full = get_ligo_data('L1', t_start - SLICE_PAD, t_end + SLICE_PAD)

    if np.isnan(h1_full.value).any() or np.isnan(l1_full.value).any():
        print("CRITICAL: Analytical bounds dropped due to hardware NaN items.")
        return

    # Isolate and filter padded frames
    h1_padded_f = manual_distinction_filter(h1_full)
    l1_padded_f = manual_distinction_filter(l1_full)

    # Trim spatial wings down cleanly to exact target limits
    h1_f = h1_padded_f.crop(t_start, t_end)
    l1_f = l1_padded_f.crop(t_start, t_end)

    # High-temporal resolution parsing (1s fftlength to extract bit-threads)
    csd = h1_f.csd(l1_f, fftlength=1, overlap=0.5)
    csd_mag = np.abs(csd.value)
    freqs = csd.frequencies.value
    
    # Restrict sub-peak extraction near the target window
    mask = (freqs > f_target - 50) & (freqs < f_target + 50)
    local_freqs = freqs[mask]
    local_mags = csd_mag[mask]
    
    sort_idx = np.argsort(local_mags)[::-1]
    
    print("\n" + "="*40)
    print("SPATIAL WAVEFRONT SUB-STRUCTURE")
    print("="*40)
    print(f"Target Center:  {f_target:.2f} Hz")
    
    top_peaks = []
    for i in range(min(5, len(sort_idx))):
        f_sub = local_freqs[sort_idx[i]]
        p_sub = local_mags[sort_idx[i]]
        print(f"Sub-Peak {i+1}: {f_sub:7.2f} Hz | Power: {p_sub:.2e}")
        if i < 3: # Keep top 3 for structured plotting highlight marks
            top_peaks.append((f_sub, p_sub))
    print("="*40 + "\n")

    # Export structured PGFPlots documentation file
    name = os.path.split(__file__)[-1]
    tex_filename = f"{name}_Wavefront_{t_start}.tex"
    export_wavefront_to_latex(freqs, csd_mag, top_peaks, f_target, start_gps, target_slice_offset, filename=tex_filename)

    # Native Verification Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, csd_mag, color='darkorchid', label='CSD Fine Structure')
    plt.scatter([p[0] for p in top_peaks], [p[1] for p in top_peaks], color='orange', marker='^', s=100, zorder=5, label='Top Sub-Peaks')
    plt.axvline(f_target, color='red', linestyle='--', label='Nariai Center')
    plt.xlim(f_target - 100, f_target + 100)
    plt.title(f"Wavefront Reconstruction: Sub-structure near {f_target:.1f}Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Correlation Magnitude")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    spatial_wavefront_scan(1266624018, 448, r_target=1.3e6)
