import os
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

def get_ligo_data(detector, start, end, cache_dir="ligo_cache"):
    """Fetches LIGO data with local HDF5 caching."""
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Construct a unique filename for this time slice
    cache_file = os.path.join(cache_dir, f"{detector}_{start}_{end}.h5")
    
    if os.path.exists(cache_file):
        print(f"Loading data from cache: {cache_file}")
        # Load the local file
        data = TimeSeries.read(cache_file)
    else:
        print(f"Cache miss. Downloading data for {detector}...")
        # Fetch from GWOSC
        data = TimeSeries.fetch_open_data(detector, start, end)
        # Save to local cache
        data.write(cache_file, overwrite=True)
        print(f"Data saved to: {cache_file}")
        
    return data

def main():
    # Configuration
    detector = 'L1'
    gps_start = 1266624018
    gps_end = 1266624618
    
    # Execution
    try:
        data = get_ligo_data(detector, gps_start, gps_end)
        
        # Plotting
        plot = data.plot(
            title=f"LIGO {detector} Strain Data",
            ylabel="Strain",
            color='darkred'
        )
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()