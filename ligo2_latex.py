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
    """Explicitly applies the distinction drive using a Butterworth manifold."""
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    filtered_data = filtfilt(b, a, data.value)
    return TimeSeries(filtered_data, sample_rate=fs, t0=data.t0)

def export_csd_to_pgfplots(frequencies, csd_mag, f_jitter_target, gps_start, gps_end, peak_freq, aperiodic_entropy, filename):
    """Generates a structured LaTeX figure containing a log-log pgfplots CSD visualization."""
    print(f"Generating LaTeX TikZ/PGFPlots file: {filename}...")
    
    # Convert GPS times to human-readable UTC strings
    utc_start = tconvert(gps_start).strftime('%Y-%m-%d %H:%M:%S UTC')
    utc_end = tconvert(gps_end).strftime('%Y-%m-%d %H:%M:%S UTC')
    
    orig_bins = len(frequencies)
    
    # --- Peak-Preserving Logarithmic Decimation ---
    # Since it's a log-log plot, we want dense sampling at low frequencies and sparse at high frequencies,
    # but we must preserve the maximum peaks so the visualization looks identical.
    target_points = 1500
    log_edges = np.logspace(np.log10(frequencies[1]), np.log10(frequencies[-1]), target_points)
    
    dec_freqs = []
    dec_mags = []
    
    # Always include the very first DC/low bin
    dec_freqs.append(frequencies[0])
    dec_mags.append(csd_mag[0])
    
    for i in range(len(log_edges) - 1):
        idx = np.where((frequencies >= log_edges[i]) & (frequencies < log_edges[i+1]))[0]
        if len(idx) > 0:
            # Pick the index that holds the maximum value within this logarithmic bucket
            max_sub_idx = idx[np.argmax(csd_mag[idx])]
            dec_freqs.append(frequencies[max_sub_idx])
            dec_mags.append(csd_mag[max_sub_idx])

    dec_freqs = np.array(dec_freqs)
    dec_mags = np.array(dec_mags)
    downsampled_bins = len(dec_freqs)
    downsample_factor = orig_bins / downsampled_bins

    # Build coordinates array
    coordinates_str = ""
    for f, m in zip(dec_freqs, dec_mags):
        if f > 0 and m > 0: # Ensure valid log coordinates
            coordinates_str += f"({f:.4f}, {m:.6e})\n"

    # LaTeX figure template with log-log axes matching matplotlib properties
    latex_template = f"""\\begin{{figure}}[htbp]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    title={{Path Identity Correlation Magnitude}},
    xlabel={{Frequency (Hz)}},
    ylabel={{Cross-Spectral Density Magnitude}},
    xmode=log,
    ymode=log,
    xmin={dec_freqs[1]:.2f}, xmax={dec_freqs[-1]:.2f},
    width=0.9\\textwidth,
    height=0.5\\textwidth,
    grid=both,
    grid style={{dashed, gray!20}},
    every axis title/.style={{font=\\bfseries}},
    legend pos=south west
]
\\addplot[color=teal, thick] coordinates {{
{coordinates_str}}};
\\addlegendentry{{CSD Magnitude}}

\\addplot[color=red, dashed, thick, no markers] coordinates {{
    ({f_jitter_target:.4f}, {np.min(dec_mags):.2e})
    ({f_jitter_target:.4f}, {np.max(dec_mags):.2e})
}};
\\addlegendentry{{Target ({f_jitter_target:.2f} Hz)}}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Cross-Spectral Density (CSD) correlation magnitude profile computed between Hanford (H1) and Livingston (L1) detectors from GPS time {gps_start} ({utc_start}) to {gps_end} ({utc_end}). The target jitter frequency corresponding to the analyzed model geometry is indicated at {f_jitter_target:.2f}~Hz, with a calculated strong spectral correlation peak identified at {peak_freq:.2f}~Hz and an aperiodic spectral entropy value of {aperiodic_entropy:.6f}~bits. The spectral profile has been mapped using peak-preserving logarithmic decimation from {orig_bins:,} original frequency bins down to {downsampled_bins:,} coordinates (a factor reduction of {downsample_factor:.1f}$\\times$) for document serialization compatibility.}}
\\label{{fig:csd_{gps_start}}}
\\end{{figure}}
"""

    with open(filename, "w") as f:
        f.write(latex_template)
    print(f"LaTeX file successfully saved to {filename} ({downsampled_bins} points).")


def analyze_aperiodic_archive(gps_start, gps_end, r_target):
    c = 299792458
    f_jitter_target = c / r_target
    fs = 4096
    
    # Add a 5 second sacrificial buffer to absorb the whiten/resample edge artifacts
    CROP_PAD = 5
    padded_start = gps_start - CROP_PAD
    padded_end = gps_end + CROP_PAD
    
    print("-" * 30)
    print("INITIALIZING APERIODIC ANALYSIS (v1.6)")
    print(f"Target Nariai Radius: {r_target:.2e} m")
    print("-" * 30)

    try:
        # 1. Fetch & Pre-process with padded windows
        h1 = get_ligo_data('H1', padded_start, padded_end).resample(fs)
        l1 = get_ligo_data('L1', padded_start, padded_end).resample(fs)

        # 2. Apply Manual Filter (The 'Distinction' Step)
        print("Applying Manual Butterworth Distinction Filter...")
        h1_filt = manual_distinction_filter(h1.whiten(), low=20, high=1200, fs=fs)
        l1_filt = manual_distinction_filter(l1.whiten(), low=20, high=1200, fs=fs)

        # Crop out edge transient buffers cleanly before processing spectral metrics
        h1_filt = h1_filt.crop(gps_start, gps_end)
        l1_filt = l1_filt.crop(gps_start, gps_end)

        # 3. Cross-Spectral Density
        csd = h1_filt.csd(l1_filt, fftlength=4, overlap=2)
        csd_mag = np.abs(csd.value)
        
        peak_idx = np.argmax(csd_mag)
        peak_freq = csd.frequencies.value[peak_idx]
        peak_val = csd_mag[peak_idx]

        # 4. Spectral Entropy (Perfect Glass Test)
        psd_h1 = h1_filt.psd(fftlength=4)
        norm_psd = psd_h1.value / np.sum(psd_h1.value)
        aperiodic_entropy = entropy(norm_psd)

        # 5. Output Results
        print("\n" + "="*40)
        print("APERIODIC SIGNATURE RESULTS")
        print("="*40)
        print(f"Aperiodic Entropy:  {aperiodic_entropy:.6f} bits")
        print(f"Target Jitter:      {f_jitter_target:.2f} Hz")
        print(f"Strongest Peak:     {peak_freq:.2f} Hz")
        print(f"Correlation Power:  {peak_val:.2e}")
        print("-" * 40)

        # 6. Export to LaTeX PGFPlots
        name = os.path.split(__file__)[-1]
        tex_filename = f"{name}_CSD_{gps_start}.tex"
        export_csd_to_pgfplots(
            csd.frequencies.value, 
            csd_mag, 
            f_jitter_target, 
            gps_start, 
            gps_end, 
            peak_freq, 
            aperiodic_entropy, 
            filename=tex_filename
        )

        # 7. Matplotlib Display
        plt.figure(figsize=(10, 5))
        plt.loglog(csd.frequencies, csd_mag, color='teal', label='CSD Magnitude')
        plt.axvline(f_jitter_target, color='red', linestyle='--', label='Target')
        plt.title("Path Identity Correlation Magnitude")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"ANALYSIS ERROR: {e}")

if __name__ == "__main__":
    analyze_aperiodic_archive(1266624018, 1266624082, 1.3e6)
