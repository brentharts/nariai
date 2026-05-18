import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.time import tconvert
from scipy.stats import entropy

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

def export_coupling_to_latex(times, h1_ents, l1_ents, start_gps, duration, r_ent, filename):
    """Generates an optimized standalone PGFPlots dual-trace entropic timeline."""
    print(f"Generating optimized LaTeX TikZ/PGFPlots entropic asset: {filename}...")
    
    end_gps = start_gps + duration
    utc_start = tconvert(start_gps).strftime('%Y-%m-%d %H:%M:%S UTC')
    utc_end = tconvert(end_gps).strftime('%Y-%m-%d %H:%M:%S UTC')

    # --- Generate Space-Separated Multi-Column Data Blocks ---
    table_rows = ""
    for t, h, l in zip(times, h1_ents, l1_ents):
        table_rows += f"{float(t):.2f} {float(h):.4f} {float(l):.4f}\n"

    duration_f = f"{float(duration):.2f}"
    
    # Dynamically scale axis padding around actual entropy extremes
    all_ents = h1_ents + l1_ents
    e_min = f"{float(np.min(all_ents) - 0.05):.2f}"
    e_max = f"{float(np.max(all_ents) + 0.05):.2f}"

    if r_ent > 0.4:
        status_str = "SHARED DISTINCTION DRIVE (UNIVERSAL)"
    elif r_ent > 0.1:
        status_str = "WEAK COUPLING (PATH IDENTITY)"
    else:
        status_str = "INDEPENDENT MANIFOLDS"

    latex_template = rf"""\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.85\textwidth,
    height=0.42\textwidth,
    title={{Universal Archive `Breathing': Entropic Fluctuations}},
    xlabel={{Time Offset (s)}},
    ylabel={{Aperiodic Entropy (bits)}},
    xmin=0,
    xmax={duration_f},
    ymin={e_min},
    ymax={e_max},
    grid=both,
    grid style={{dashed, gray!10}},
    legend pos=north east,
    legend columns=2,
    legend style={{nodes={{scale=0.8, transform shape}}}},
    every axis title/.style={{font=\bfseries\small}}
]

% 1. H1 Entropic Timeline Trace
\addplot[color=teal, thick, no markers] table [x=time, y=h1] {{
time h1 l1
{table_rows}}};
\addlegendentry{{H1 Entropy}}

% 2. L1 Entropic Timeline Trace
\addplot[color=red, thick, no markers] table [x=time, y=l1] {{
time h1 l1
{table_rows}}};
\addlegendentry{{L1 Entropy}}

\end{{axis}}
\end{{tikzpicture}}
\caption{{Parallel informational tracking profiling global spectral entropy fluctuations across a {duration}-second macro-window spanning from GPS {start_gps} ({utc_start}) to {end_gps} ({utc_end}). Structural evaluations isolate an exact entropic coupling correlation metric of $R = {r_ent:.4f}$. Current system manifold state validation: \textbf{{{status_str}}}.}}
\label{{fig:entropic_coupling_{start_gps}}}
\end{{figure}}
"""

    with open(filename, "w") as f:
        f.write(latex_template)
    print(f"LaTeX file successfully saved to {filename}")

def robust_entropy_coupling(start_gps, total_duration):
    fs = 4096
    slice_size = 64
    
    # Defensive data wing buffer to eliminate local filter start-up noise
    SLICE_PAD = 4
    
    print(f"\n--- INITIATING ROBUST ENTROPY COUPLING ({total_duration}s) ---")
    
    # Load foundational raw baseline blocks backed by safe padding extensions
    h1_full = get_ligo_data('H1', start_gps - SLICE_PAD, start_gps + total_duration + SLICE_PAD)
    l1_full = get_ligo_data('L1', start_gps - SLICE_PAD, start_gps + total_duration + SLICE_PAD)
    
    h1_ents = []
    l1_ents = []
    valid_times = []
    
    for t in range(0, total_duration - slice_size, slice_size):
        try:
            t_start = start_gps + t
            t_end = t_start + slice_size
            
            # Anomaly Filter: Reject segments with severe tracking/instrumentation dropouts
            h1_check = h1_full.crop(t_start, t_end)
            l1_check = l1_full.crop(t_start, t_end)
            if np.isnan(h1_check.value).any() or np.isnan(l1_check.value).any() or np.all(h1_check.value == 0):
                continue
                
            # EXECUTION FIX: Whiten the buffered array segment first to shield the analytics from edge noise
            h_white_padded = h1_full.crop(t_start - SLICE_PAD, t_end + SLICE_PAD).whiten()
            l_white_padded = l1_full.crop(t_start - SLICE_PAD, t_end + SLICE_PAD).whiten()
            
            # Crop down cleanly to clear the corrupted boundary transients
            h_crop = h_white_padded.crop(t_start, t_end)
            l_crop = l_white_padded.crop(t_start, t_end)
            
            # Compute Power Spectral Density profiles
            h_psd = h_crop.psd(fftlength=4)
            l_psd = l_crop.psd(fftlength=4)
            
            # Isolate relative spectral entropy
            h_ent = entropy(h_psd.value / np.sum(h_psd.value))
            l_ent = entropy(l_psd.value / np.sum(l_psd.value))
            
            if np.isfinite(h_ent) and np.isfinite(l_ent):
                h1_ents.append(h_ent)
                l1_ents.append(l_ent)
                valid_times.append(t)
        except Exception as e:
            continue

    if not valid_times:
        print("\nCRITICAL RUN FAILURE: Zero data slices passed information-theoretic quality thresholds.")
        return

    # Quantify corporate Pearson Correlation Coefficient between the entropic arrays
    if len(h1_ents) > 1:
        ent_correlation = np.corrcoef(h1_ents, l1_ents)[0, 1]
    else:
        ent_correlation = 0.0

    print("\n" + "="*40)
    print("CLEANED ENTROPIC COUPLING RESULTS")
    print("="*40)
    print(f"Valid Slices:        {len(h1_ents)}")
    print(f"Entropy Correlation: {ent_correlation:.4f}")
    
    if ent_correlation > 0.4:
        print("STATUS: SHARED DISTINCTION DRIVE (UNIVERSAL)")
    elif ent_correlation > 0.1:
        print("STATUS: WEAK COUPLING (PATH IDENTITY)")
    else:
        print("STATUS: INDEPENDENT MANIFOLDS")
    print("="*40 + "\n")

    # Export structured PGFPlots documentation file
    name = os.path.split(__file__)[-1]
    tex_filename = f"{name}_EntropicCoupling_{start_gps}.tex"
    export_coupling_to_latex(valid_times, h1_ents, l1_ents, start_gps, total_duration, ent_correlation, filename=tex_filename)

    # Native Verification Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(valid_times, h1_ents, label='H1 Entropy', color='teal', alpha=0.8)
    plt.plot(valid_times, l1_ents, label='L1 Entropy', color='orchid', alpha=0.8)
    plt.title("Universal Archive 'Breathing': Entropic Fluctuations")
    plt.ylabel("Aperiodic Entropy (bits)")
    plt.xlabel("Time (s)")
    plt.legend(ncols=2)
    plt.grid(alpha=0.2)
    plt.show()

if __name__ == "__main__":
    robust_entropy_coupling(1266624018, 2048)
