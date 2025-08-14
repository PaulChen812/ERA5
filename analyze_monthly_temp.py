import xarray as xr
import numpy as np
import os
import glob

class analyze_monthly_temp:
    def __init__(self, directory):
        self.directory = directory
        self.files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".nc")])

    def __call__(self):
        datasets = [xr.open_dataset(f)["t2m"] - 273.15 for f in self.files]
        combined = xr.concat(datasets, dim="time")

        # Calculate statistics
        mean = combined.mean().item()
        std = combined.std().item()
        var = combined.var().item()
        min_val = combined.min().item()
        max_val = combined.max().item()
        median = float(np.median(combined.values))

        print(f"Monthly Temperature Statistics for {self.directory}:")
        print(f"Mean: {mean:.2f} °C")
        print(f"Standard Deviation: {std:.2f} °C")
        print(f"Variance: {var:.2f} °C²")
        print(f"Min: {min_val:.2f} °C")
        print(f"Max: {max_val:.2f} °C")
        print(f"Median: {median:.2f} °C")

if __name__ == "__main__":
    analyze = analyze_monthly_temp("/global/cfs/cdirs/m3638/ERA5_Data/2m_temperature_daily/2025/01")
    analyze()
