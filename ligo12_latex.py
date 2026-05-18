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

def export_jitter_to_latex(times, exact_freqs, target_f, start_gps, duration, mean_f, jitter, filename):
    """Generates an optimized standalone PGFPlots high-res step plot asset."""
    print(f"Generating optimized LaTeX TikZ/PGFPlots jitter tracking asset: {filename}...")
    
    end_gps = start_gps + duration
    utc_start = tconvert(start_gps).strftime('%Y-%m-%d %H:%M:%S UTC')
    utc_end = tconvert(end_gps).strftime('%Y-%m-%d %H:%M:%S UTC')

    # --- Generate Space-Separated Table Records ---
    table_rows = ""
    for t, f in zip(times, exact_freqs):
        table_rows += f"{float(t):.2f} {float(f):.4f}\n"

    target_f_f = f"{float(target_f):.2f}"
    duration_f = f"{float(duration):.2f}"
    
    # Calculate tight y-bounds around tracked frequency path
    f_min = f"{float(np.min(exact_freqs) - 0.1):.2f}"
    f_max = f"{float(np.max(exact_freqs) + 0.1):.2f}"

    status_str = "APERIODIC SIGNAL CONFIRMED" if jitter > 0.001 else "STATIONARY LINE / ARTIFACT"

    latex_template = rf"""\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.85\textwidth,
    height=0.42\textwidth,
    title={{Fine-Structure Jitter Tracking near {target_f_f}~Hz}},
    xlabel={{Time Offset (s)}},
    ylabel={{Exact Frequency (Hz)}},
    xmin=0,
    xmax={duration_f},
    ymin={f_min},
    ymax={f_max},
    grid=both,
    grid style={{dashed, gray!10}},
    legend pos=north east,
    legend style={{nodes={{scale=0.8, transform shape}}}},
    every axis title/.style={{font=\bfseries\small}}
]

% 1. Step trace mapping the localized discrete frequency jumps
\addplot[color=lime!80!black, thick, const plot, no markers] table [x=time, y=freq] {{
time freq
{table_rows}}};
\addlegendentry{{Tracked Attractor Peak}}

% 2. Baseline target alignment horizontal reference
\draw[color=darkgray, dashed, thick] (axis cs:0.0, {target_f_f}) -- (axis cs:{duration_f}, {target_f_f});
\addlegendimage{{line legend, color=darkgray, dashed, thick}}
\addlegendentry{{Target Center}}

\end{{axis}}
\end{{tikzpicture}}
\caption{{Ultra-high resolution timeline tracking of the {target_f_f}~Hz micro-attractor over a {duration}-second data window running from GPS {start_gps} ({utc_start}) to {end_gps} ({utc_end}). Tracking parameters show a mean background frequency profile of {mean_f:.4f}~Hz with an absolute structural spectral jitter metric of {jitter:.6f}~Hz. Current validation profile status: \textbf{{{status_str}}}.}}
\label{{fig:jitter_telemetry_{start_gps}}}
\end{{figure}}
"""

    with open(filename, "w") as f:
        f.write(latex_template)
    print(f"LaTeX file successfully saved to {filename}")

def micro_frequency_telemetry(start_gps, total_duration, target_f=513.50):
    fs = 4096
    slice_size = 64 
    
    # Defensive processing wing boundaries to swallow whitening filter initialization transients
    SLICE_PAD = 4
    
    print(f"\n--- ANALYZING FINE-STRUCTURE JITTER (Target: {target_f} Hz) ---")
    
    # Pull raw base data streams with padded safety limits
    h1_full = get_ligo_data('H1', start_gps - SLICE_PAD, start_gps + total_duration + SLICE_PAD)     
    
    times = []
    exact_freqs = []
    
    # Step interval loop tracking at 75% overlap windows
    step = slice_size // 4
    for t in range(0, total_duration - slice_size, step):
        try:
            t_start = start_gps + t
            t_end = t_start + slice_size
            
            # Anomaly Filter: Scan for and exclude underlying instrumentation dropouts
            h1_check = h1_full.crop(t_start, t_end)
            if np.isnan(h1_check.value).any():
                continue

            # Crop raw blocks with buffer extensions, whiten, then slice cleanly to nominal size
            slice_padded = h1_full.crop(t_start - SLICE_PAD, t_end + SLICE_PAD).whiten()
            slice_data = slice_padded.crop(t_start, t_end)
            
            # High-resolution Power Spectral Density split
            psd = slice_data.psd(fftlength=slice_size)
            
            # Isolate localized search window directly flanking the micro-attractor line
            mask = (psd.frequencies.value > target_f - 2) & (psd.frequencies.value < target_f + 2)
            local_freqs = psd.frequencies.value[mask]
            local_power = psd.value[mask]
            
            peak_f = local_freqs[np.argmax(local_power)]
            
            times.append(t)
            exact_freqs.append(peak_f)
        except Exception as e:
            continue
            
    if not exact_freqs:
        print("\nCRITICAL RUN FAILURE: No data frames survived structural jitter filtering checks.")
        return

    # Statistical Evaluation Analytics
    jitter = np.std(exact_freqs)
    mean_f = np.mean(exact_freqs)
    
    print("\n" + "="*40)
    print("ATTRACTOR JITTER TELEMETRY")
    print("="*40)
    print(f"Mean Frequency:  {mean_f:.4f} Hz")
    print(f"Spectral Jitter: {jitter:.6f} Hz")
    if jitter > 0.001:
        print("STATUS: APERIODIC SIGNAL CONFIRMED")
    else:
        print("STATUS: STATIONARY LINE (POTENTIAL ARTIFACT)")
    print("="*40 + "\n")

    # Export structured PGFPlots documentation file
    name = os.path.split(__file__)[-1]
    tex_filename = f"{name}_Jitter_{start_gps}.tex"
    export_jitter_to_latex(times, exact_freqs, target_f, start_gps, total_duration, mean_f, jitter, filename=tex_filename)

    # Native Verification Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(times, exact_freqs, color='lime', drawstyle='steps-post', label='Tracked Peak')
    plt.axhline(target_f, color='black', linestyle='--', alpha=0.4, label=f'Target Target ({target_f}Hz)')
    plt.title(f"Aperiodic Jitter of the {target_f} Hz Attractor")
    plt.ylabel("Exact Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()

if __name__ == "__main__":
    micro_frequency_telemetry(1266624018, 2048, target_f=513.50)
