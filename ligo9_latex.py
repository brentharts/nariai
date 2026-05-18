import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.time import tconvert
from scipy.signal import coherence

def get_ligo_data(detector, start, end, cache_dir="ligo_cache"):
    if not os.path.exists(cache_dir): os.makedirs(cache_dir)
    cache_file = os.path.join(cache_dir, f"{detector}_{start}_{end}.h5")
    if os.path.exists(cache_file):
        return TimeSeries.read(cache_file)
    print(f"Cache miss. Downloading {detector} ({end-start}s)...")
    data = TimeSeries.fetch_open_data(detector, start, end, sample_rate=4096)
    data.write(cache_file, overwrite=True)
    return data

def export_coherence_to_latex(band_freqs, band_coh, f_target, start_gps, offset, mean_coh, peak_f, peak_val, filename):
    """Generates an optimized standalone PGFPlots coherence tracking line asset."""
    print(f"Generating optimized LaTeX TikZ/PGFPlots coherence asset: {filename}...")
    
    t_slice = start_gps + offset
    utc_slice = tconvert(t_slice).strftime('%Y-%m-%d %H:%M:%S UTC')

    # --- Generate Space-Separated Table Records ---
    table_rows = ""
    for f, c in zip(band_freqs, band_coh):
        table_rows += f"{f:.2f} {c:.4f}\n"

    f_target_f = f"{float(f_target):.2f}"
    f_min = f"{float(np.min(band_freqs)):.2f}"
    f_max = f"{float(np.max(band_freqs)):.2f}"

    latex_template = rf"""\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.85\textwidth,
    height=0.42\textwidth,
    title={{Manifold Coherence: Path Identity Verification}},
    xlabel={{Frequency (Hz)}},
    ylabel={{Coherence (0--1)}},
    xmin={f_min},
    xmax={f_max},
    ymin=0,
    ymax=1.0,
    grid=both,
    grid style={{dashed, gray!10}},
    legend pos=north east,
    legend style={{nodes={{scale=0.8, transform shape}}}},
    every axis title/.style={{font=\bfseries\small}}
]

% 1. Continuous Magnitude Squared Coherence Plot
\addplot[color=crimson, thick, no markers] table [x=freq, y=coh] {{
freq coh
{table_rows}}};
\addlegendentry{{H1-L1 Coherence}}

% 2. Vertical reference line mapping the theoretical target frequency
\draw[color=black, dashed, thick] (axis cs:{f_target_f}, \pgfkeysvalueof{{/pgfplots/ymin}}) -- (axis cs:{f_target_f}, \pgfkeysvalueof{{/pgfplots/ymax}});
\addlegendimage{{line legend, color=black, dashed, thick}}
\addlegendentry{{Theoretical Nariai}}

\end{{axis}}
\end{{tikzpicture}}
\caption{{Magnitude squared cross-coherence spectrum resolved at $1\text{{ Hz}}$ resolution over a 32-second timeline tracking window at T+{offset}s [GPS {t_slice}] ({utc_slice}). The localized band profiles a mean cross-coherence background metric of {mean_coh:.4f} with a dominant coherent path alignment spike peaking at {peak_val:.4f} localized precisely at {peak_f:.2f}~Hz relative to the theoretical prediction framework centered at {f_target_f}~Hz.}}
\label{{fig:coherence_scan_{t_slice}}}
\end{{figure}}
"""

    with open(filename, "w") as f:
        f.write(latex_template)
    print(f"LaTeX file successfully saved to {filename}")

def analyze_phase_coherence(start_gps, target_offset, r_target=1.3e6):
    c = 299792458
    f_target = c / r_target
    fs = 4096
    
    # Defensive window buffer to insulate the whitening transform boundary edges
    SLICE_PAD = 4
    
    t_start = start_gps + target_offset
    t_end = t_start + 32
    
    print(f"\n--- MEASURING MANIFOLD COHERENCE (T+{target_offset}s) ---")
    
    # Load raw target block with surrounding buffer wings
    h1_full = get_ligo_data('H1', t_start - SLICE_PAD, t_end + SLICE_PAD)
    l1_full = get_ligo_data('L1', t_start - SLICE_PAD, t_end + SLICE_PAD)

    if np.isnan(h1_full.value).any() or np.isnan(l1_full.value).any():
        print("CRITICAL: Analytical bounds dropped due to hardware NaN items.")
        return

    # Whiten the padded arrays to absorb filter edge corruptions
    h1_white_padded = h1_full.whiten()
    l1_white_padded = l1_full.whiten()

    # Crop out pristine uncompromised time frames
    h1_f = h1_white_padded.crop(t_start, t_end)
    l1_f = l1_white_padded.crop(t_start, t_end)

    # Calculate Coherence (1Hz temporal segment windows)
    freqs, coh = coherence(h1_f.value, l1_f.value, fs=fs, nperseg=fs)

    # Focus explicitly on the structural Nariai Band
    mask = (freqs > 150) & (freqs < 350)
    band_freqs = freqs[mask]
    band_coh = coh[mask]
    
    peak_coh_idx = np.argmax(band_coh)
    mean_band_coh = np.mean(band_coh)
    peak_freq_val = band_freqs[peak_coh_idx]
    peak_coh_val = band_coh[peak_coh_idx]
    
    print("\n" + "="*40)
    print("PATH IDENTITY COHERENCE RESULTS")
    print("="*40)
    print(f"Peak Coherence: {peak_coh_val:.4f}")
    print(f"At Frequency:   {peak_freq_val:.2f} Hz")
    print(f"Mean Band Coh:  {mean_band_coh:.4f}")
    print("="*40 + "\n")

    # Export structured PGFPlots documentation file
    name = os.path.split(__file__)[-1]
    tex_filename = f"{name}_Coherence_{t_start}.tex"
    export_coherence_to_latex(
        band_freqs, band_coh, f_target, start_gps, target_offset,
        mean_band_coh, peak_freq_val, peak_coh_val, filename=tex_filename
    )

    # Native Verification Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(band_freqs, band_coh, color='crimson', label='H1-L1 Coherence')
    plt.axvline(f_target, color='black', linestyle='--', label=f'Theoretical Nariai ({f_target:.1f}Hz)')
    plt.title("Manifold Coherence: Path Identity Verification")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence (0-1)")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    analyze_phase_coherence(1266624018, 448, r_target=1.3e6)
