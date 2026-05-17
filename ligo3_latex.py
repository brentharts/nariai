import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.time import tconvert
from scipy.stats import entropy
from scipy.signal import butter, filtfilt

def get_ligo_data(detector, start, end, cache_dir="ligo_cache"):
    if not os.path.exists(cache_dir): 
        os.makedirs(cache_dir)
    cache_file = os.path.join(cache_dir, f"{detector}_{start}_{end}.h5")
    if os.path.exists(cache_file):
        return TimeSeries.read(cache_file)
    data = TimeSeries.fetch_open_data(detector, start, end, sample_rate=4096)
    data.write(cache_file, overwrite=True)
    return data

def manual_distinction_filter(data, low=20, high=1200, fs=4096):
    """Applies a Butterworth bandpass filter to a pre-whitened data manifold."""
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    white = data.whiten()
    filtered_data = filtfilt(b, a, white.value)
    return TimeSeries(filtered_data, sample_rate=fs, t0=data.t0)

def export_scan_to_pgfplots(times, entropies, peaks, f_target, gps_start, gps_end, filename):
    """Generates a LaTeX figure matching a dual y-axis matplotlib visualization layout."""
    print(f"Generating dual-axis LaTeX TikZ/PGFPlots file: {filename}...")
    
    # Convert GPS times to human-readable UTC strings
    utc_start = tconvert(gps_start).strftime('%Y-%m-%d %H:%M:%S UTC')
    utc_end = tconvert(gps_end).strftime('%Y-%m-%d %H:%M:%S UTC')
    
    mean_ent = np.mean(entropies)
    peak_var = np.var(peaks)

    # Build coordinate string lists
    entropy_coords = "".join([f"({t}, {e:.6f})\n" for t, e in zip(times, entropies)])
    peak_coords = "".join([f"({t}, {p:.2f})\n" for t, p in zip(times, peaks)])

    # Double y-axis LaTeX template using pgfplots 'axis y line=left/right' engine
    latex_template = f"""
\\definecolor{{teal}}{{RGB}}{{0,128,128}}
\\definecolor{{darkred}}{{RGB}}{{139,0,0}}
\\begin{{figure}}[htbp]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    title={{Temporal Drift Metrics and Frequency Stability}},
    xlabel={{Time (seconds from GPS {gps_start})}},
    ylabel={{Spectral Entropy (bits)}},
    xmin={times[0]}, xmax={times[-1]},
    width=0.9\\textwidth,
    height=0.5\\textwidth,
    grid=major,
    grid style={{dashed, gray!20}},
    axis y line=left,
    axis x line=bottom,
    y tick label style={{/pgf/number format/fixed, /pgf/number format/precision=4}},
    every axis title/.style={{font=\\bfseries}},
    legend pos=north west
]
\\addplot[color=teal, thick, mark=*] coordinates {{
{entropy_coords}}};
\\addlegendentry{{Spectral Entropy}}
\\end{{axis}}

\\begin{{axis}}[
    xmin={times[0]}, xmax={times[-1]},
    width=0.9\\textwidth,
    height=0.5\\textwidth,
    ylabel={{Strongest Peak Frequency (Hz)}},
    axis y line=right,
    axis x line=none,
    legend pos=north east
]
\\addplot[color=darkred, dashed, thick, mark=square*] coordinates {{
{peak_coords}}};
\\addlegendentry{{Strongest CSD Peak}}

\\addplot[color=red, dotted, ultra thick, no markers] coordinates {{
    ({times[0]}, {f_target:.2f})
    ({times[-1]}, {f_target:.2f})
}};
\\addlegendentry{{Predicted Target ({f_target:.2f} Hz)}}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Temporal evolution of signal structural changes evaluated across a {gps_end - gps_start}-second frame from GPS time {gps_start} ({utc_start}) to {gps_end} ({utc_end}). Metrics are extracted using non-overlapping sequential windows. The primary axis tracks the evolution of spectral entropy (Mean: {mean_ent:.4f}~bits), while the secondary axis maps the stability of the dominant Cross-Spectral Density peak against the theoretical model framework prediction line at {f_target:.2f}~Hz (Observed Peak Variance: {peak_var:.2f}~$\\text{{Hz}}^2$). All sample streams are cleanly isolated from edge conditioning transients.}}
\\label{{fig:temporal_scan_{gps_start}}}
\\end{{figure}}
"""

    with open(filename, "w") as f:
        f.write(latex_template)
    print(f"LaTeX file successfully saved to {filename} ({len(times)} structural updates).")

def scan_temporal_archive(start_gps, duration, slice_size=16, r_target=1.3e6):
    c = 299792458
    f_target = c / r_target
    fs = 4096
    
    # Sacrificial pad to execute high-fidelity conditioning prior to chopping up blocks
    CROP_PAD = 8 
    
    times = []
    entropies = []
    peaks = []

    print(f"SCANNING ARCHIVE: {duration}s window in {slice_size}s increments")
    
    # 1. Fetch large contiguous block with defensive pads to clear filter edge ringings
    h1_full = get_ligo_data('H1', start_gps - CROP_PAD, start_gps + duration + CROP_PAD)
    l1_full = get_ligo_data('L1', start_gps - CROP_PAD, start_gps + duration + CROP_PAD)

    # 2. Condition the entire global timeseries once to block slice-boundary artifacts
    print("Pre-conditioning global data streams...")
    h1_conditioned = manual_distinction_filter(h1_full, fs=fs)
    l1_conditioned = manual_distinction_filter(l1_full, fs=fs)

    # Loop strictly within the valid uncorrupted region
    for t in range(0, duration, slice_size):
        try:
            t_start = start_gps + t
            t_end = t_start + slice_size
            
            if t_end > start_gps + duration:
                break
            
            # Extract safe sub-slices from the cleanly conditioned parent series
            h1_s = h1_conditioned.crop(t_start, t_end)
            l1_s = l1_conditioned.crop(t_start, t_end)

            # CSD and Entropy
            csd = h1_s.csd(l1_s, fftlength=4, overlap=2)
            psd = h1_s.psd(fftlength=4)
            
            ent = entropy(psd.value / np.sum(psd.value))
            peak_f = csd.frequencies.value[np.argmax(np.abs(csd.value))]
            
            times.append(t)
            entropies.append(ent)
            peaks.append(peak_f)
            
            print(f"T+{t:02d}s | Entropy: {ent:.4f} | Peak: {peak_f:.2f} Hz")
            
        except Exception as e:
            print(f"Slice processing error at T+{t}: {e}")
            continue

    # Final Summary for copy-paste
    print("\n" + "="*40)
    print("TEMPORAL DRIFT SUMMARY")
    print("="*40)
    print(f"Mean Entropy: {np.mean(entropies):.6f}")
    print(f"Peak Variance: {np.var(peaks):.2f}")
    print(f"F_Jitter Predicted: {f_target:.2f} Hz")
    print("="*40)

    # 3. Export to LaTeX PGFPlots Dual Axis Diagram
    name = os.path.split(__file__)[-1]
    tex_filename = f"{name}_Scan_{start_gps}.tex"
    export_scan_to_pgfplots(times, entropies, peaks, f_target, start_gps, start_gps + duration, tex_filename)

    # 4. Added Matplotlib Dual Axis Display Block
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Left axis setup: Entropy
    color = 'teal'
    ax1.set_xlabel(f'Time (seconds from GPS {start_gps})')
    ax1.set_ylabel('Spectral Entropy (bits)', color=color)
    ax1.plot(times, entropies, color=color, marker='o', linewidth=2, label='Entropy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Right axis setup: Peak Frequency
    ax2 = ax1.twinx()  
    color = 'darkred'
    ax2.set_ylabel('Strongest Peak Frequency (Hz)', color=color)
    ax2.plot(times, peaks, color=color, linestyle='--', marker='s', linewidth=2, label='Peak Freq')
    ax2.axhline(f_target, color='red', linestyle=':', linewidth=2, label='Target Value')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Temporal Drift Metrics and Frequency Stability")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Scanning a 64-second window to look for "Archive Persistence" with 8-second increments
    scan_temporal_archive(1266624018, 64, slice_size=8)
