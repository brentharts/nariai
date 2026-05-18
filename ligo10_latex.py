import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.time import tconvert
from scipy.signal import coherence

def get_deep_archive(detector, start, duration, cache_dir="ligo_cache"):
    if not os.path.exists(cache_dir): 
        os.makedirs(cache_dir)
    cache_file = os.path.join(cache_dir, f"{detector}_{start}_{start+duration}.h5")
    if os.path.exists(cache_file):
        print(f"Loading {detector} Deep Archive from cache...")
        return TimeSeries.read(cache_file)
    print(f"Downloading {detector} Deep Archive ({duration}s)...")
    data = TimeSeries.fetch_open_data(detector, start, start+duration, sample_rate=4096)
    data.write(cache_file, overwrite=True)
    return data

def export_deep_telemetry_to_latex(times, freqs, coherences, f_target, start_gps, duration, filename):
    """Generates an optimized PGFPlots representation of long-duration coherent drift."""
    print(f"Generating optimized LaTeX TikZ/PGFPlots deep archive asset: {filename}...")
    
    end_gps = start_gps + duration
    utc_start = tconvert(start_gps).strftime('%Y-%m-%d %H:%M:%S UTC')
    utc_end = tconvert(end_gps).strftime('%Y-%m-%d %H:%M:%S UTC')

    # --- Generate Space-Separated Table Records ---
    table_rows = ""
    for t, f, c in zip(times, freqs, coherences):
        table_rows += f"{float(t):.2f} {float(f):.2f} {float(c):.4f}\n"

    min_coh, max_coh = (np.min(coherences), np.max(coherences)) if coherences else (0, 1)
    f_target_f = f"{float(f_target):.2f}"
    duration_f = f"{float(duration):.2f}"

    latex_template = rf"""\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.85\textwidth,
    height=0.45\textwidth,
    title={{Deep Archive Path Identity Persistence}},
    xlabel={{Time Offset (s)}},
    ylabel={{Coherent Frequency (Hz)}},
    grid=both,
    grid style={{dashed, gray!10}},
    scatter,
    point meta=explicit,
    colorbar,
    colorbar style={{
        title={{\scriptsize Coherence}},
        title style={{at={{(0.5,1.05)}}, anchor=south}},
        yticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=3}}
    }},
    point meta min={min_coh:.4f},
    point meta max={max_coh:.4f},
    colormap name=magma,
    every axis title/.style={{font=\bfseries\small}}
]

\addplot[
    only marks,
    mark=*,
    mark size=1.8pt,
    opacity=0.75
] table [x=time, y=freq, meta=coh] {{
time freq coh
{table_rows}}};

% Theoretical structural reference line
\draw[color=cyan, dashed, thick] (axis cs:0.0, {f_target_f}) -- (axis cs:{duration_f}, {f_target_f});

\end{{axis}}
\end{{tikzpicture}}
\caption{{Macro-scale timeline scan over a comprehensive {duration}-second integration window running from GPS {start_gps} ({utc_start}) to {end_gps} ({utc_end}). Individual points track the dominant cross-coherence peak frequencies verified across overlapping sliding evaluation blocks. The horizontal dashed line maps the theoretical geometric stability track fixed at {f_target_f}~Hz.}}
\label{{fig:deep_telemetry_{start_gps}}}
\end{{figure}}
"""

    with open(filename, "w") as f:
        f.write(latex_template)
    print(f"LaTeX file successfully saved to {filename}")

def deep_manifold_scan(start_gps, total_duration, slice_size=128, r_target=1.3e6):
    c_light = 299792458
    f_target = c_light / r_target
    fs = 4096
    
    # Defensive sliding padding bounds to isolate crop edge distortions
    SLICE_PAD = 4
    
    print(f"\n--- INITIATING DEEP ARCHIVE INTEGRATION ({total_duration}s) ---")
    
    # Fetch full underlying telemetry framework with safety padding margins
    h1_big = get_deep_archive('H1', start_gps - SLICE_PAD, total_duration + (SLICE_PAD * 2))
    l1_big = get_deep_archive('L1', start_gps - SLICE_PAD, total_duration + (SLICE_PAD * 2))

    global_times = []
    global_coherence = []
    global_freqs = []

    # Sliding window execution with 50% step overlap for continuity mapping
    step = slice_size // 2
    for t in range(0, total_duration - slice_size, step):
        try:
            t_start = start_gps + t
            t_end = t_start + slice_size
            
            # Anomaly Filter: Detect and drop hardware instrumentation dropouts early
            h1_check = h1_big.crop(t_start, t_end)
            l1_check = l1_big.crop(t_start, t_end)
            if np.isnan(h1_check.value).any() or np.isnan(l1_check.value).any():
                continue

            # Process padded slice blocks to sweep filter edge transients
            h1_s = h1_big.crop(t_start - SLICE_PAD, t_end + SLICE_PAD).whiten()
            l1_s = l1_big.crop(t_start - SLICE_PAD, t_end + SLICE_PAD).whiten()
            
            # Trim away boundary contamination zones cleanly
            h1_clean = h1_s.crop(t_start, t_end)
            l1_clean = l1_s.crop(t_start, t_end)

            # High-resolution frequency-coherence matrix split
            f, c = coherence(h1_clean.value, l1_clean.value, fs=fs, nperseg=fs * 2)
            
            # Filter cross-coherence calculations tightly inside the explicate domain
            mask = (f > 20) & (f < 1000)
            local_freqs = f[mask]
            local_coh = c[mask]
            
            peak_idx = np.argmax(local_coh)
            
            global_times.append(t)
            global_coherence.append(local_coh[peak_idx])
            global_freqs.append(local_freqs[peak_idx])
            
            if t % 256 == 0:
                print(f"Progress: {t:4d}s | Max Coh: {global_coherence[-1]:.4f} at {global_freqs[-1]:.2f} Hz")
        except Exception as e:
            continue

    if not global_times:
        print("\nCRITICAL RUN FAILURE: No data subsets survived analytical filter routines.")
        return

    # Mathematical Summary Traces
    max_idx = np.argmax(global_coherence)
    max_coh_val = global_coherence[max_idx]
    max_coh_freq = global_freqs[max_idx]
    mean_stability = np.mean(global_coherence)

    print("\n" + "="*40)
    print("DEEP ARCHIVE COHERENCE SUMMARY")
    print("="*40)
    print(f"Max Global Coherence: {max_coh_val:.4f}")
    print(f"Frequency at Max:    {max_coh_freq:.2f} Hz")
    print(f"Mean Stability:      {mean_stability:.4f}")
    print("="*40 + "\n")

    # Export structural LaTeX document asset
    name = os.path.split(__file__)[-1]
    tex_filename = f"{name}_DeepScan_{start_gps}.tex"
    export_deep_telemetry_to_latex(
        global_times, global_freqs, global_coherence,
        f_target, start_gps, total_duration, filename=tex_filename
    )

    # Native Verification Plotting
    plt.figure(figsize=(12, 5))
    plt.scatter(global_times, global_freqs, c=global_coherence, cmap='inferno', s=30, edgecolors='none')
    plt.colorbar(label='Coherence Magnitude')
    plt.axhline(f_target, color='cyan', linestyle='--', alpha=0.6, label=f'Nariai Target ({f_target:.1f} Hz)')
    plt.title("Deep Archive Path Identity Persistence")
    plt.xlabel("Time Offset (s)")
    plt.ylabel("Coherent Frequency (Hz)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    robust_analysis_gps = 1266624018 
    deep_manifold_scan(robust_analysis_gps, 2048, slice_size=128, r_target=1.3e6)
