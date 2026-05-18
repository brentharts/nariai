import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.time import tconvert

def get_ligo_data(detector, start, end, cache_dir="ligo_cache"):
    if not os.path.exists(cache_dir): 
        os.makedirs(cache_dir)
    cache_file = os.path.join(cache_dir, f"{detector}_{start}_{end}.h5")
    if os.path.exists(cache_file):
        return TimeSeries.read(cache_file)
    print(f"Cache miss. Downloading {detector} ({end-start}s)...")
    data = TimeSeries.fetch_open_data(detector, start, end, sample_rate=4096)
    data.write(cache_file, overwrite=True)
    return data

def export_correlation_to_latex(times, h1_deltas, l1_deltas, target_f, start_gps, duration, r_corr, filename):
    """Generates an optimized standalone PGFPlots dual-trace delta tracking asset."""
    print(f"Generating optimized LaTeX TikZ/PGFPlots correlation asset: {filename}...")
    
    end_gps = start_gps + duration
    utc_start = tconvert(start_gps).strftime('%Y-%m-%d %H:%M:%S UTC')
    utc_end = tconvert(end_gps).strftime('%Y-%m-%d %H:%M:%S UTC')

    # --- Generate Unified Space-Separated Multi-Column Data Blocks ---
    table_rows = ""
    for t, h, l in zip(times, h1_deltas, l1_deltas):
        table_rows += f"{float(t):.2f} {float(h):.4f} {float(l):.4f}\n"

    target_f_f = f"{float(target_f):.2f}"
    duration_f = f"{float(duration):.2f}"
    
    # Calculate symmetry for y-bounds based on max absolute delta excursions
    max_excursion = max(np.max(np.abs(h1_deltas)), np.max(np.abs(l1_deltas)))
    y_bound = f"{float(max_excursion + 0.15):.2f}"

    status_str = "UNIVERSAL PATH IDENTITY CONFIRMED" if r_corr > 0.5 else "LOCALIZED FLUCTUATIONS"

    latex_template = rf"""\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.85\textwidth,
    height=0.42\textwidth,
    title={{Synchronized Aperiodic Jitter (H1 vs. L1)}},
    xlabel={{Time Offset (s)}},
    ylabel={{$\Delta$ Frequency (Hz)}},
    xmin=0,
    xmax={duration_f},
    ymin=-{y_bound},
    ymax={y_bound},
    grid=both,
    grid style={{dashed, gray!10}},
    legend pos=north east,
    legend style={{nodes={{scale=0.8, transform shape}}}},
    every axis title/.style={{font=\bfseries\small}}
]

% 1. H1 Delta Frequency Step Trace
\addplot[color=cyan, thick, const plot, no markers] table [x=time, y=h1] {{
time h1 l1
{table_rows}}};
\addlegendentry{{H1 Jitter}}

% 2. L1 Delta Frequency Step Trace
\addplot[color=magenta, thick, const plot, no markers] table [x=time, y=l1] {{
time h1 l1
{table_rows}}};
\addlegendentry{{L1 Jitter}}

% 3. Zero-offset horizontal center line
\draw[color=gray, dashed] (axis cs:0.0, 0.0) -- (axis cs:{duration_f}, 0.0);

\end{{axis}}
\end{{tikzpicture}}
\caption{{Synchronized dual-detector tracking of the {target_f_f}~Hz attractor over a {duration}-second data window from GPS {start_gps} ({utc_start}) to {end_gps} ({utc_end}). Profiles display a joint Pearson correlation coefficient of $R = {r_corr:.4f}$ between the independent tracking hubs. Current network verification status: \textbf{{{status_str}}}.}}
\label{{fig:jitter_correlation_{start_gps}}}
\end{{figure}}
"""

    with open(filename, "w") as f:
        f.write(latex_template)
    print(f"LaTeX file successfully saved to {filename}")

def non_local_jitter_correlation(start_gps, total_duration, target_f=513.9):
    fs = 4096
    slice_size = 64
    step = slice_size // 4
    
    # Defensive data wing buffer to eliminate local filter start-up noise
    SLICE_PAD = 4
    
    print(f"\n--- VERIFYING NON-LOCAL JITTER CORRELATION ({total_duration}s) ---")
    
    # Load foundational data frameworks with safe surrounding wings
    h1_full = get_ligo_data('H1', start_gps - SLICE_PAD, start_gps + total_duration + SLICE_PAD)
    l1_full = get_ligo_data('L1', start_gps - SLICE_PAD, start_gps + total_duration + SLICE_PAD)
    
    times = []
    h1_freqs = []
    l1_freqs = []
    
    for t in range(0, total_duration - slice_size, step):
        try:
            t_start = start_gps + t
            t_end = t_start + slice_size
            
            # Anomaly filter: identify dropouts across both components simultaneously
            h1_check = h1_full.crop(t_start, t_end)
            l1_check = l1_full.crop(t_start, t_end)
            if np.isnan(h1_check.value).any() or np.isnan(l1_check.value).any():
                continue

            # Process padded slices through the whitening filter, then trim away wings
            h1_padded = h1_full.crop(t_start - SLICE_PAD, t_end + SLICE_PAD).whiten()
            l1_padded = l1_full.crop(t_start - SLICE_PAD, t_end + SLICE_PAD).whiten()
            
            h1_slice = h1_padded.crop(t_start, t_end)
            l1_slice = l1_padded.crop(t_start, t_end)
            
            # Generate high-resolution Power Spectral Density footprints
            h_psd = h1_slice.psd(fftlength=slice_size)
            l_psd = l1_slice.psd(fftlength=slice_size)
            
            # Bound search mask near target attractor window
            mask = (h_psd.frequencies.value > target_f - 3) & (h_psd.frequencies.value < target_f + 3)
            local_freqs = h_psd.frequencies.value[mask]
            
            h1_freqs.append(local_freqs[np.argmax(h_psd.value[mask])])
            l1_freqs.append(local_freqs[np.argmax(l_psd.value[mask])])
            times.append(t)
        except Exception as e:
            continue

    if not times:
        print("\nCRITICAL RUN FAILURE: No data frames survived non-local tracking checks.")
        return

    # Quantify the Pearson Correlation Coefficient of the tracked jitter lines
    correlation = np.corrcoef(h1_freqs, l1_freqs)[0, 1]
    
    print("\n" + "="*40)
    print("NON-LOCAL ARCHIVE CORRELATION")
    print("="*40)
    print(f"Jitter Correlation: {correlation:.4f}")
    if correlation > 0.5:
        print("STATUS: UNIVERSAL PATH IDENTITY CONFIRMED")
    else:
        print("STATUS: LOCALIZED FLUCTUATIONS")
    print("="*40 + "\n")

    # Isolate delta frequencies away from base center for tracking storage
    h1_deltas = np.array(h1_freqs) - target_f
    l1_deltas = np.array(l1_freqs) - target_f

    # Export structural LaTeX document asset
    name = os.path.split(__file__)[-1]
    tex_filename = f"{name}_Correlation_{start_gps}.tex"
    export_correlation_to_latex(times, h1_deltas, l1_deltas, target_f, start_gps, total_duration, correlation, filename=tex_filename)

    # Native Verification Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(times, h1_deltas, label='H1 Jitter', color='cyan', alpha=0.7, drawstyle='steps-post')
    plt.plot(times, l1_deltas, label='L1 Jitter', color='magenta', alpha=0.7, drawstyle='steps-post')
    plt.axhline(0.0, color='black', linestyle='--', alpha=0.3)
    plt.title("Synchronized Aperiodic Jitter (H1 vs L1)")
    plt.ylabel("Delta Freq (Hz)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    non_local_jitter_correlation(1266624018, 2048, target_f=513.9)
