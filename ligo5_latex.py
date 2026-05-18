import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.time import tconvert
from scipy.stats import entropy
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

def export_distribution_plots_to_latex(peaks, time_indices, entropies, f_target, start_gps, duration, mean_ent, lock_rate, filename):
    """Generates a side-by-side subfigure via pgfplots groupplots to track peak distributions and drift."""
    print(f"Generating optimized LaTeX TikZ/PGFPlots multi-panel asset: {filename}...")
    
    end_gps = start_gps + duration
    utc_start = tconvert(start_gps).strftime('%Y-%m-%d %H:%M:%S UTC')
    utc_end = tconvert(end_gps).strftime('%Y-%m-%d %H:%M:%S UTC')
    
    # --- Generate Flat Data Table Rows for Native Hist Parsing ---
    hist_table_rows = ""
    for p in peaks:
        hist_table_rows += f"{float(p):.2f}\n"

    # Calculate safe vertical limit for the histogram line plot reference line
    counts, _ = np.histogram(peaks, bins=15)
    hist_ymax = float(np.max(counts) * 1.2) if len(counts) > 0 else 10.0

    # --- Generate Scatter Plot Table Rows ---
    scatter_table_rows = ""
    for t, p, e in zip(time_indices, peaks, entropies):
        scatter_table_rows += f"{float(t):.2f} {float(p):.2f} {float(e):.4f}\n"

    # Get colorbar range boundaries
    min_ent, max_ent = (np.min(entropies), np.max(entropies)) if entropies else (0, 1)

    # Convert numeric parameters to explicit string floats
    f_target_f = f"{float(f_target):.2f}"
    duration_f = f"{float(duration):.2f}"

    # Raw f-string protects LaTeX backslashes while cleanly interpolating metrics
    latex_template = rf"""\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{
        group size=2 by 1,
        horizontal sep=2.2cm
    }},
    width=0.48\textwidth,
    height=0.42\textwidth,
    grid=both,
    grid style={{dashed, gray!10}},
    every axis title/.style={{font=\bfseries\small}}
]

% --- PANEL A: PERSISTENCE HISTOGRAM ---
\nextgroupplot[
    title={{Distribution of Jitter Peaks}},
    xlabel={{Frequency (Hz)}},
    ylabel={{Detections}},
    ymin=0,
    legend pos=north east,
    legend style={{nodes={{scale=0.7, transform shape}}}}
]
% Let PGFPlots calculate bins natively from raw column values to prevent bar scrambling
\addplot[
    ybar interval,
    fill=teal!60,
    draw=teal!90,
    hist={{bins=15}}
] table [y=val] {{
val
{hist_table_rows}}};
\addlegendentry{{Peak Counts}}

\draw[color=crimson, dashed, thick] (axis cs:{f_target_f}, 0.0) -- (axis cs:{f_target_f}, {hist_ymax:.2f});
\addlegendimage{{line legend, color=crimson, dashed, thick}}
\addlegendentry{{Nariai Limit}}

% --- PANEL B: SCATTER DISTRIBUTION VS TIME ---
\nextgroupplot[
    title={{Jitter Frequency vs. Time}},
    xlabel={{Time (s)}},
    ylabel={{Peak Frequency (Hz)}},
    scatter,
    point meta=explicit,
    colorbar,
    colorbar style={{
        title={{\scriptsize Entropy}},
        title style={{at={{(0.5,1.05)}}, anchor=south}},
        yticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=3}}
    }},
    point meta min={min_ent:.4f},
    point meta max={max_ent:.4f},
    colormap/viridis
]
\addplot[
    only marks,
    mark=*,
    mark size=2.5pt,
    opacity=0.85
] table [x=time, y=freq, meta=ent] {{
time freq ent
{scatter_table_rows}}};

% Draw horizontal reference line cleanly in background coordinates
\draw[color=crimson, dashed, thick] (axis cs:0.0, {f_target_f}) -- (axis cs:{duration_f}, {f_target_f});

\end{{groupplot}}
\end{{tikzpicture}}
\caption{{Aperiodic stability profile over a {duration}-second data stream starting at GPS {start_gps} ({utc_start}) to {end_gps} ({utc_end}). \textbf{{Left:}} Persistence frequency distribution across verified evaluation windows relative to the theoretical Nariai limit of {f_target:.2f}~Hz. \textbf{{Right:}} Evolution of instantaneous peak frequency drift mapped across active timelines, where individual markers are continuously shaded by localized spectral entropy metrics. The historical track displays a structural baseline alignment mean entropy of {mean_ent:.4f}~bits and an exact phase lock-in metric of {lock_rate:.1f}\% inside a $\pm2\%$ tracking envelope.}}
\label{{fig:distribution_profile_{start_gps}}}
\end{{figure}}
"""

    with open(filename, "w") as f:
        f.write(latex_template)
    print(f"LaTeX file successfully saved to {filename}")

def robust_archive_analysis(start_gps, duration, slice_size=32, r_target=1.3e6):
    c = 299792458
    f_target = c / r_target
    fs = 4096
    
    # Defensive spatial buffer to protect edge integrity against whitening/filtering artifacts
    SLICE_PAD = 4
    
    entropies = []
    peaks = []
    time_indices = []

    print(f"\n--- ANALYZING ARCHIVE DISTRIBUTION ({duration}s) ---")
    
    # 1. Fetch the raw background strain block with outer safety wings
    h1_full = get_ligo_data('H1', start_gps - SLICE_PAD, start_gps + duration + SLICE_PAD)
    l1_full = get_ligo_data('L1', start_gps - SLICE_PAD, start_gps + duration + SLICE_PAD)

    # 2. Insulated Processing Loop
    for t in range(0, duration - slice_size, slice_size):
        try:
            t_start = start_gps + t
            t_end = t_start + slice_size
            
            # Extract standard raw slices to evaluate baseline signal continuity
            h1_check = h1_full.crop(t_start, t_end)
            l1_check = l1_full.crop(t_start, t_end)
            
            # Drop individual slice if an underlying instrument dropout (NaN) occurs
            if np.isnan(h1_check.value).any() or np.isnan(l1_check.value).any():
                print(f"Slice {t:03d}s | SKIPPED due to internal hardware dropout anomaly (NaN items encountered).")
                continue

            # Apply buffer wings prior to whitening to consume transition edge transients
            h1_s = h1_full.crop(t_start - SLICE_PAD, t_end + SLICE_PAD)
            l1_s = l1_full.crop(t_start - SLICE_PAD, t_end + SLICE_PAD)

            # Isolate and filter the padded block
            h1_f_padded = manual_distinction_filter(h1_s)
            l1_f_padded = manual_distinction_filter(l1_s)
            
            # Completely trim away contaminated pad boundaries before doing arithmetic
            h1_f = h1_f_padded.crop(t_start, t_end)
            l1_f = l1_f_padded.crop(t_start, t_end)

            # Cross-Spectral Density & Power Spectral Density
            csd = h1_f.csd(l1_f, fftlength=4, overlap=2)
            csd_mag = np.abs(csd.value)
            psd = h1_f.psd(fftlength=4)
            
            ent = entropy(psd.value / np.sum(psd.value))
            peak_f = csd.frequencies.value[np.argmax(csd_mag)]
            
            # Store data points if they pass validation checks
            if np.isfinite(ent) and peak_f > 0:
                entropies.append(ent)
                peaks.append(peak_f)
                time_indices.append(t)
                print(f"Slice {t:03d}s | Entropy: {ent:.4f} | Peak Peak: {peak_f:.2f} Hz")
            else:
                print(f"Slice {t:03d}s | SKIPPED due to unstable spectral processing values.")
        except Exception as e:
            print(f"Skipping slice at T+{t}s due to execution fault: {e}")
            continue

    # Exit cleanly if no active configurations survive filtering metrics
    if not peaks:
        print("\nCRITICAL RUN FAILURE: No data intervals survived anomaly clearing routines.")
        return

    # 3. Comprehensive Global Metrics
    mean_ent = np.nanmean(entropies)
    locked = [p for p in peaks if abs(p - f_target) < (0.02 * f_target)]
    lock_rate = (len(locked) / len(peaks)) * 100

    print("\n" + "="*40)
    print("CLEANED APERIODIC RESULTS")
    print("="*40)
    print(f"Mean Entropy:      {mean_ent:.6f} bits")
    print(f"Nariai Target:     {f_target:.2f} Hz")
    print(f"Lock-in Rate:      {lock_rate:.1f}%")
    print(f"Peak Count:        {len(peaks)} active slices")
    print("="*40 + "\n")

    # 4. Export structured side-by-side LaTeX document asset
    name = os.path.split(__file__)[-1]
    tex_filename = f"{name}_Distributions_{start_gps}.tex"
    export_distribution_plots_to_latex(
        peaks, time_indices, entropies, 
        f_target, start_gps, duration, 
        mean_ent, lock_rate, filename=tex_filename
    )

    # 5. Native Verification Plotting
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(peaks, bins=15, color='teal', edgecolor='black', alpha=0.7)
    plt.axvline(f_target, color='crimson', linestyle='--', label=f'Nariai Limit ({f_target:.1f}Hz)')
    plt.title("Distribution of Jitter Peaks")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Detections")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(time_indices, peaks, c=entropies, cmap='viridis', s=50, alpha=0.85, edgecolor='none')
    plt.axhline(f_target, color='crimson', linestyle='--', label=f'Target ({f_target:.1f}Hz)')
    plt.colorbar(label='Aperiodic Entropy')
    plt.title("Jitter Frequency vs. Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Peak Frequency (Hz)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    GPS_START = 1266624018 
    DURATION = 512
    R_NARIAI = 1.3e6
    
    robust_archive_analysis(GPS_START, DURATION, slice_size=32, r_target=R_NARIAI)
