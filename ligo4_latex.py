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
        print(f"Loading {detector} from cache...")
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

def export_long_csd_to_pgfplots(frequencies, csd_mag, f_target, start_gps, duration, entropies, filename):
    """Generates a structured log-log pgfplots LaTeX figure from the decimated final CSD slice."""
    print(f"Generating optimized LaTeX TikZ/PGFPlots file: {filename}...")
    
    end_gps = start_gps + duration
    utc_start = tconvert(start_gps).strftime('%Y-%m-%d %H:%M:%S UTC')
    utc_end = tconvert(end_gps).strftime('%Y-%m-%d %H:%M:%S UTC')
    
    orig_bins = len(frequencies)
    # Use nanmean to completely insulate metric strings from invalid points
    mean_entropy = np.nanmean(entropies)
    
    # --- Peak-Preserving Logarithmic Decimation ---
    target_points = 1500
    log_edges = np.logspace(np.log10(frequencies[1]), np.log10(frequencies[-1]), target_points)
    
    dec_freqs = [frequencies[0]]
    dec_mags = [csd_mag[0]]
    
    for i in range(len(log_edges) - 1):
        idx = np.where((frequencies >= log_edges[i]) & (frequencies < log_edges[i+1]))[0]
        if len(idx) > 0:
            max_sub_idx = idx[np.argmax(csd_mag[idx])]
            dec_freqs.append(frequencies[max_sub_idx])
            dec_mags.append(csd_mag[max_sub_idx])

    dec_freqs = np.array(dec_freqs)
    dec_mags = np.array(dec_mags)
    downsampled_bins = len(dec_freqs)
    downsample_factor = orig_bins / downsampled_bins

    coordinates_str = ""
    for f, m in zip(dec_freqs, dec_mags):
        if f > 0 and m > 0:
            coordinates_str += f"({f:.4f}, {m:.6e})\n"

    latex_template = f"""\\begin{{figure}}[htbp]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    title={{Aperiodic Archive Analysis: {duration}s Window}},
    xlabel={{Frequency (Hz)}},
    ylabel={{Correlation Magnitude}},
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
\\addplot[color=blue, opacity=0.8, thick] coordinates {{
{coordinates_str}}};
\\addlegendentry{{Correlation (CSD)}}

\\addplot[color=red, dashed, thick, no markers] coordinates {{
    ({f_target:.4f}, {np.min(dec_mags):.2e})
    ({f_target:.4f}, {np.max(dec_mags):.2e})
}};
\\addlegendentry{{Predicted Nariai Jitter ({f_target:.1f} Hz)}}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Cross-Spectral Density (CSD) correlation magnitude profile across an extended window of {duration}~seconds from GPS time {start_gps} ({utc_start}) to {end_gps} ({utc_end}). The graph illustrates a representative snapshot taken from the final evaluation slice against a baseline global mean spectral entropy of {mean_entropy:.4f}~bits. The theoretical Nariai jitter target line is marked at {f_target:.2f}~Hz. Frequency-domain elements are compressed via a peak-preserving logarithmic decimation filter from {orig_bins:,} to {downsampled_bins:,} coordinates ({downsample_factor:.1f}$\\times$ reduction) to support memory-efficient LaTeX compilation without sacrificing structural alignment. Edge conditioning transients were completely excised via internal padding prior to spectral calculation, and sections containing hardware dropout anomalies were dynamically filtered out of global averages.}}
\\label{{fig:long_csd_{start_gps}}}
\\end{{figure}}
"""

    with open(filename, "w") as f:
        f.write(latex_template)
    print(f"LaTeX file successfully saved to {filename} ({downsampled_bins} data points written).")

def long_window_archive_scan(start_gps, duration, slice_size=32, r_target=1.3e6):
    c = 299792458
    f_target = c / r_target
    fs = 4096
    
    # Padded buffer for individual slices to protect them from edge artifacts
    SLICE_PAD = 4
    
    entropies = []
    peaks = []

    print(f"\n--- STARTING LONG-WINDOW SCAN ({duration}s) ---")
    
    # 1. Fetch the entire block with extra padding at the outer boundaries
    h1_full = get_ligo_data('H1', start_gps - SLICE_PAD, start_gps + duration + SLICE_PAD)
    l1_full = get_ligo_data('L1', start_gps - SLICE_PAD, start_gps + duration + SLICE_PAD)

    # 2. Iterative Analysis Loop
    for t in range(0, duration - slice_size, slice_size):
        try:
            t_start = start_gps + t
            t_end = t_start + slice_size
            
            # Crop raw slices first to safely verify detector data health
            h1_check = h1_full.crop(t_start, t_end)
            l1_check = l1_full.crop(t_start, t_end)
            
            # Dynamic filtering check: check for any detector locks/drops (NaNs) inside the slice
            if np.isnan(h1_check.value).any() or np.isnan(l1_check.value).any():
                print(f"Slice {t:03d}s | SKIPPED: Internal hardware dropout anomaly detected (NaN entries present).")
                continue

            # Crop with an extra padding buffer on both sides for safe filtering
            h1_s = h1_full.crop(t_start - SLICE_PAD, t_end + SLICE_PAD)
            l1_s = l1_full.crop(t_start - SLICE_PAD, t_end + SLICE_PAD)

            # Whiten and filter the padded slices
            h1_f_padded = manual_distinction_filter(h1_s)
            l1_f_padded = manual_distinction_filter(l1_s)
            
            # Trim the padding away to remove edge transients before calculation
            h1_f = h1_f_padded.crop(t_start, t_end)
            l1_f = l1_f_padded.crop(t_start, t_end)

            # Cross-Spectral Density
            csd = h1_f.csd(l1_f, fftlength=4, overlap=2)
            csd_mag = np.abs(csd.value)
            
            # Entropy check
            psd = h1_f.psd(fftlength=4)
            ent = entropy(psd.value / np.sum(psd.value))
            
            peak_f = csd.frequencies.value[np.argmax(csd_mag)]
            
            # Double check scalar outputs to be absolutely certain no NaNs get stored
            if np.isnan(ent) or np.isnan(peak_f):
                print(f"Slice {t:03d}s | SKIPPED: Spectral calculation produced an unstable value.")
                continue
                
            entropies.append(ent)
            peaks.append(peak_f)
            
            print(f"Slice {t:03d}s | Entropy: {ent:.4f} | Peak: {peak_f:.2f} Hz")
        except Exception as e:
            print(f"Skipping slice at T+{t}s due to error: {e}")
            continue

    # Verify we gathered valid uncorrupted snapshots before building figures
    if len(entropies) == 0:
        print("\nCRITICAL FAILURE: All data segments in this window contained hardware dropouts.")
        return

    # 3. Final Summary Output (using nan-safe array reductions)
    print("\n" + "="*40)
    print("LONG-WINDOW ARCHIVE SUMMARY")
    print("="*40)
    print(f"Total Duration:   {duration}s")
    print(f"Global Mean Ent:  {np.nanmean(entropies):.6f}")
    print(f"Frequency Drift:  {np.nanmin(peaks):.1f} Hz to {np.nanmax(peaks):.1f} Hz")
    print(f"Jitter Predict:   {f_target:.2f} Hz")
    print("="*40 + "\n")

    # 4. Export to LaTeX PGFPlots
    name = os.path.split(__file__)[-1]
    tex_filename = f"{name}_LongCSD_{start_gps}.tex"
    export_long_csd_to_pgfplots(
        csd.frequencies.value, 
        csd_mag, 
        f_target, 
        start_gps, 
        duration, 
        entropies, 
        filename=tex_filename
    )

    # 5. Global Plotting
    plt.figure(figsize=(12, 6))
    plt.loglog(csd.frequencies, csd_mag, color='blue', alpha=0.8, label='Correlation (CSD)')
    plt.axvline(f_target, color='red', linestyle='--', label=f'Predicted Nariai Jitter ({f_target:.1f}Hz)')
    plt.title(f"Aperiodic Archive Analysis: {duration}s Window\n(Mean Entropy: {np.nanmean(entropies):.4f})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Correlation Magnitude")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

if __name__ == "__main__":
    GPS_START = 1266624018 
    DURATION = 512
    R_NARIAI = 1.3e6
    
    long_window_archive_scan(GPS_START, DURATION, slice_size=32, r_target=R_NARIAI)
