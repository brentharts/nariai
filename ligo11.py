import numpy as np
import matplotlib.pyplot as plt

def clean_and_analyze_spectre(times, freqs, coherences, f_target):
    # 1. Filter out the 'Implicate Collapses' (NaNs)
    mask = np.isfinite(coherences)
    c_times = np.array(times)[mask]
    c_freqs = np.array(freqs)[mask]
    c_cohs = np.array(coherences)[mask]

    # 2. Ratio Analysis
    ratios = c_freqs / f_target
    
    print("\n" + "="*40)
    print("SPECTRE RATIO VALIDATION")
    print("="*40)
    print(f"Mean Path Coherence: {np.mean(c_cohs):.4f}")
    print(f"Primary Attractor:   {c_freqs[np.argmax(c_cohs)]:.2f} Hz")
    print(f"Attractor Ratio:     {np.max(ratios):.4f} (Target: 2.236 for sqrt(5))")
    
    # Check for recurring ratios
    unique_freqs, counts = np.unique(np.round(c_freqs, 0), return_counts=True)
    persistence = unique_freqs[counts > 1]
    print(f"Persistent Modes:    {persistence} Hz")
    print("="*40)

    # 3. Visualization of the 'Aperiodic Ladder'
    plt.figure(figsize=(10, 5))
    plt.scatter(c_times, ratios, c=c_cohs, cmap='plasma', s=100, label='Detection Slices')
    plt.axhline(1.0, color='cyan', linestyle='--', label='Nariai Base')
    plt.axhline(np.sqrt(5), color='gold', linestyle=':', label='sqrt(5) Scaling')
    plt.title("Aperiodic Scaling of the Path Identity")
    plt.ylabel("Ratio (f / f_nariai)")
    plt.xlabel("Time Offset (s)")
    plt.colorbar(label='Coherence')
    plt.legend()
    plt.show()

# Use the data from your deep scan
t = [0, 256, 512, 768, 1024, 1280, 1536, 1792]
f = [435.50, 20.50, 513.50, 513.50, 731.00, 513.50, 337.50, 393.50]
c = [0.0708, np.nan, 0.0717, 0.0895, 0.0526, 0.1909, 0.0712, 0.0634]

clean_and_analyze_spectre(t, f, c, 230.61)