import xarray as xr
import numpy as np
from era_climatology import CLIMATOLOGY_MEAN, MONTHLY_CLIMATOLOGY

def calculate_variance_std_nc(nc_file: str, var_name: str = "Temperature"):
    ds = xr.open_dataset(nc_file)
    data = ds[var_name]

    if "Year" not in ds or "Month" not in ds:
        raise ValueError("NetCDF file must contain Year and Month variables")
    
    years = ds["Year"].values
    months = ds["Month"].values

    results = {}


    overall_diff = data.values - CLIMATOLOGY_MEAN
    results["overall"] = {
        "variance": float(np.var(overall_diff, ddof=0)),
        "std_dev": float(np.std(overall_diff, ddof=0)),
    }


    for m in range(1, 13):
        mask = months == m
        if np.any(mask):
            diff = data.values[mask] - MONTHLY_CLIMATOLOGY[m]
            results[m] = {
                "variance": float(np.var(diff, ddof=0)),
                "std_dev": float(np.std(diff, ddof=0)),
            }

    return results


if __name__ == "__main__":
    nc_file = "ERA_US_Monthly_mean.nc"
    stats = calculate_variance_std_nc(nc_file)

    print("=== Variance & Standard Deviation Compared to Climatology ===")
    print(f"Overall: variance={stats['overall']['variance']:.3f}, std={stats['overall']['std_dev']:.3f}")
    for month in range(1, 13):
        v = stats[month]["variance"]
        s = stats[month]["std_dev"]
        print(f"Month {month:02d}: variance={v:.3f}, std={s:.3f}")
