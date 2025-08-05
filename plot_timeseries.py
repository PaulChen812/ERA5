import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from dask.diagnostics import ProgressBar

def main(directory, lat, lon, output_file="point_timeseries.png"):
    print(f"Opening files in {directory}")
    print(f"Target coordinate: lat={lat}, lon={lon}")

    ds = xr.open_mfdataset(
        os.path.join(directory, "*.nc"),
        combine="by_coords",
        parallel=True,
        chunks={"time": 8},
        engine="netcdf4"
    )

    t2m = ds["t2m"].astype("float64")
    t2m_celsius = t2m - 273.15  # decode scale/offset, convert to °C

    # Find nearest lat/lon in grid
    point_data = t2m_celsius.sel(
        latitude=lat,
        longitude=lon,
        method="nearest"
    )

    # Compute time and temperature
    with ProgressBar():
        times = point_data["time"].values
        temps = point_data.compute().values

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(times, temps, color="firebrick", linewidth=1.5)
    plt.title(f"2m Temperature Time Series at ({lat:.2f}°, {lon:.2f}°)", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Temperature (°C)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    print(f"Plot saved as: {output_file}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python plot_point_timeseries.py <directory> <lat> <lon> [output_file]")
    else:
        directory = sys.argv[1]
        lat = float(sys.argv[2])
        lon = float(sys.argv[3])
        output_file = sys.argv[4] if len(sys.argv) > 4 else "point_timeseries.png"
        main(directory, lat, lon, output_file)

