import os
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.time import tconvert

def get_ligo_data(detector, start, end, cache_dir="ligo_cache"):
    """Fetches LIGO data with local HDF5 caching."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_file = os.path.join(cache_dir, f"{detector}_{start}_{end}.h5")
    
    if os.path.exists(cache_file):
        print(f"Loading data from cache: {cache_file}")
        data = TimeSeries.read(cache_file)
    else:
        print(f"Cache miss. Downloading data for {detector}...")
        data = TimeSeries.fetch_open_data(detector, start, end)
        data.write(cache_file, overwrite=True)
        print(f"Data saved to: {cache_file}")
        
    return data

def export_to_pgfplots(data, original_sample_rate, original_size, detector, start, end, filename="ligo_plot.tex"):
    """Generates a structured LaTeX figure containing a downsampled pgfplots axis, free of edge transients."""
    print(f"Generating optimized LaTeX TikZ/PGFPlots file: {filename}...")
    
    # Convert GPS times to human-readable UTC strings
    utc_start = tconvert(start).strftime('%Y-%m-%d %H:%M:%S UTC')
    utc_end = tconvert(end).strftime('%Y-%m-%d %H:%M:%S UTC')
    
    # Metrics for the caption
    downsampled_size = len(data)
    downsample_factor = original_size / downsampled_size
    downsampled_rate = data.sample_rate.value

    # Extract times relative to the initial absolute request start time
    times = data.times.value - start 
    strains = data.value

    # Build the data coordinates string
    coordinates_str = ""
    for t, s in zip(times, strains):
        coordinates_str += f"({t:.6f}, {s:.6e})\n"

    # LaTeX template matching the matplotlib style properties closely
    latex_template = f"""
\\definecolor{{darkred}}{{RGB}}{{139,0,0}} % Defines the exact darkred line color
\\begin{{figure}}[htbp]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    title={{LIGO {detector} Strain Data}},
    xlabel={{Time (seconds from GPS {start})}},
    ylabel={{Strain}},
    xmin={times[0]}, xmax={times[-1]},
    width=0.9\\textwidth,
    height=0.5\\textwidth,
    grid=major,
    grid style={{dashed, gray!30}},
    no markers,
    every axis title/.style={{font=\\bfseries}},
]
\\addplot[color=darkred, thick] coordinates {{
{coordinates_str}}};
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{LIGO {detector} gravitational wave strain data spanning from GPS time {start} ({utc_start}) to {end} ({utc_end}). The data has been digitally downsampled from its native sampling rate of {original_sample_rate}~Hz ({original_size:,} original samples) to {downsampled_rate}~Hz ({downsampled_size:,} samples), representing a factor of {downsample_factor:.1f}$\\times$ reduction for graphical rendering. Initial filter settling transients have been cropped out for visual clarity.}}
\\label{{fig:ligo_{detector}_{start}}}
\\end{{figure}}
"""

    with open(filename, "w") as f:
        f.write(latex_template)
    print(f"LaTeX file successfully saved to {filename} ({downsampled_size} points).")

def main():
    # Configuration
    detector = 'L1'
    gps_start = 1266624018
    gps_end = 1266624618
    
    # Target sampling rate for the LaTeX plot
    TARGET_LATEX_RATE = 4 
    CROP_PAD_SECONDS = 5  # Time to discard at edges to get rid of IIR filter ringing
    
    # Execution
    try:
        # To handle cropping safely, we grab a slightly wider window from the cache/GWOSC 
        # so our final cropped data matches your exact requested gps_start and gps_end boundaries.
        padded_start = gps_start - CROP_PAD_SECONDS
        padded_end = gps_end + CROP_PAD_SECONDS
        
        padded_data = get_ligo_data(detector, padded_start, padded_end)
        
        # 1. Standard Matplotlib Display (using the high-res requested window)
        # Crop back to original request window for the clean display
        display_data = padded_data.crop(gps_start, gps_end)
        orig_rate = display_data.sample_rate.value
        orig_size = len(display_data)
        
        plot = display_data.plot(
            title=f"LIGO {detector} Strain Data",
            ylabel="Strain",
            color='darkred'
        )
        
        # 2. Resample and then Crop the padded artifacts away
        print(f"Resampling data from {orig_rate} Hz down to {TARGET_LATEX_RATE} Hz...")
        latex_data = padded_data.resample(TARGET_LATEX_RATE)
        
        # Now slice away the corrupted padded regions
        latex_data = latex_data.crop(gps_start, gps_end)
        
        # Determine the file naming convention
        name = os.path.split(__file__)[-1]
        tex_filename = f"{name}_{detector}_{gps_start}.tex"
        
        # Export the pristine, resampled dataset
        export_to_pgfplots(latex_data, orig_rate, orig_size, detector, gps_start, gps_end, filename=tex_filename)
        
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
