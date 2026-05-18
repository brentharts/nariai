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

def export_telemetry_to_latex(results, f_target, start_gps, duration, mean_ent, lock_rate, filename):
    """Generates a standalone, scannable PGFPlots telemetry scatter panel asset."""
    print(f"Generating optimized LaTeX TikZ/PGFPlots telemetry asset: {filename}...")
    
    end_gps = start_gps + duration
    utc_start = tconvert(start_gps).strftime('%Y-%m-%d %H:%M:%S UTC')
    utc_end = tconvert(end_gps).strftime('%Y-%m-%d %H:%M:%S UTC')

    # --- Generate Space-Separated Table Records ---
    table_data_rows = ""
    entropies = [r['e'] for r in results]
    for r in results:
        table_data_rows += f"{float(r['t']):.2f} {float(r['f']):.2f} {float(r['e']):.4f}\n"

    min_ent, max_ent = (np.min(entropies), np.max(entropies)) if entropies else (0, 1)
    f_target_f = f"{float(f_target):.2f}"
    duration_f = f"{float(duration):.2f}"

    latex_template = rf"""\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.85\textwidth,
    height=0.45\textwidth,
    title={{Aperiodic Jitter Telemetry: Frequency vs. Time}},
    xlabel={{Time Offset (s)}},
    ylabel={{Peak Correlation Frequency (Hz)}},
    grid=both,
    grid style={{dashed, gray!10}},
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
    colormap name=magma,
    every axis title/.style={{font=\bfseries\small}}
]

\addplot[
    only marks,
    mark=*,
    mark size=2.5pt,
    opacity=0.85
] table [x=time, y=freq, meta=ent] {{
time freq ent
{table_data_rows}}};

% Reference line isolated from scatter parser
\draw[color=cyan, dashed, thick] (axis cs:0.0, {f_target_f}) -- (axis cs:{duration_f}, {f_target_f});

\end{{axis}}
\end{{tikzpicture}}
\caption{{Aperiodic telemetry timeline over {duration} seconds [{start_gps} to {end_gps}] ({utc_start}). Individual points track instantaneous peak correlation frequencies shaded by local spectral entropy. Baseline parameters show a mean entropy value of {mean_ent:.4f}~bits and an exact phase lock-in metric of {lock_rate:.2f}\% against the Nariai target of {f_target_f}~Hz.}}
\label{{fig:telemetry_scan_{start_gps}}}
\end{{figure}}
"""

    with open(filename, "w") as f:
        f.write(latex_template)
    print(f"LaTeX file successfully saved to {filename}")

def full_telemetry_scan(start_gps, duration, slice_size=32, r_target=1.3e6):
    c = 299792458
    f_target = c / r_target
    fs = 4096
    
    # Defensive spatial buffer to protect edge integrity against whitening transients
    SLICE_PAD = 4
    results = []

    print(f"\n--- INITIATING FULL ARCHIVE TELEMETRY ({duration}s) ---")
    
    # Fetch raw data background blocks with outer safety buffers
    h1_full = get_ligo_data('H1', start_gps - SLICE_PAD, start_gps + duration + SLICE_PAD)
    l1_full = get_ligo_data('L1', start_gps - SLICE_PAD, start_gps + duration + SLICE_PAD)

    for t in range(0, duration - slice_size, slice_size):
        try:
            t_start = start_gps + t
            t_end = t_start + slice_size
            
            # Extract basic raw slices to evaluate for dropouts
            h1_check = h1_full.crop(t_start, t_end)
            l1_check = l1_full.crop(t_start, t_end)
            
            if np.isnan(h1_check.value).any() or np.isnan(l1_check.value).any():
                print(f"T+{t:03d}s | SKIPPED due to internal hardware dropout anomaly (NaN item detected).")
                continue

            # Apply buffer wings prior to whitening to swallow filter edge transients
            h1_s = h1_full.crop(t_start - SLICE_PAD, t_end + SLICE_PAD)
            l1_s = l1_full.crop(t_start - SLICE_PAD, t_end + SLICE_PAD)

            h1_f_padded = manual_distinction_filter(h1_s)
            l1_f_padded = manual_distinction_filter(l1_s)
            
            # Trim away contaminated boundary buffers
            h1_f = h1_f_padded.crop(t_start, t_end)
            l1_f = l1_f_padded.crop(t_start, t_end)

            # Cross-Spectral Density & Power Spectral Density calculations
            csd = h1_f.csd(l1_f, fftlength=4, overlap=2)
            csd_mag = np.abs(csd.value)
            psd = h1_f.psd(fftlength=4)
            
            ent = entropy(psd.value / np.sum(psd.value))
            peak_f = csd.frequencies.value[np.argmax(csd_mag)]
            
            if np.isfinite(ent) and peak_f > 0:
                is_locked = abs(peak_f - f_target) < (0.02 * f_target)
                status = "[LOCKED]" if is_locked else "        "
                print(f"T+{t:03d}s | {status} Freq: {peak_f:7.2f} Hz | Entropy: {ent:.6f}")
                results.append({'t': t, 'f': peak_f, 'e': ent, 'locked': is_locked})
            else:
                print(f"T+{t:03d}s | SKIPPED due to unstable spectral processing values.")
        except Exception as e:
            print(f"Skipping slice at T+{t}s due to execution fault: {e}")
            continue

    if not results:
        print("\nCRITICAL RUN FAILURE: No data intervals survived telemetry anomaly filters.")
        return

    # Summary Statistics
    entropies = [r['e'] for r in results]
    peaks = [r['f'] for r in results]
    lock_count = sum(1 for r in results if r['locked'])
    mean_ent = np.mean(entropies)
    lock_rate = (lock_count / len(results)) * 100
    
    print("\n" + "="*40)
    print("FINAL ARCHIVE TELEMETRY SUMMARY")
    print("="*40)
    print(f"Global Mean Entropy: {mean_ent:.6f} bits")
    print(f"Total Slices:        {len(results)}")
    print(f"Nariai Lock-ins:     {lock_count}")
    print(f"Lock-in Rate:        {lock_rate:.2f}%")
    print("="*40 + "\n")

    # Export structural LaTeX document asset
    name = os.path.split(__file__)[-1]
    tex_filename = f"{name}_Telemetry_{start_gps}.tex"
    export_telemetry_to_latex(results, f_target, start_gps, duration, mean_ent, lock_rate, filename=tex_filename)

    # Plotting Logic
    plt.figure(figsize=(10, 6))
    plt.scatter([r['t'] for r in results], peaks, c=entropies, cmap='magma', s=100, edgecolors='black')
    plt.axhline(f_target, color='cyan', linestyle='--', label=f'Nariai Target ({f_target:.1f}Hz)')
    plt.title("Aperiodic Jitter Telemetry: Frequency vs. Time")
    plt.xlabel("Time Offset (s)")
    plt.ylabel("Peak Correlation Frequency (Hz)")
    plt.colorbar(label="Aperiodic Entropy (bits)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    robust_analysis_gps = 1266624018 
    full_telemetry_scan(robust_analysis_gps, 512, slice_size=32, r_target=1.3e6)
