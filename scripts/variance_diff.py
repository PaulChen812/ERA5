import xarray as xr
from pathlib import Path
import numpy as np

# === Input files ===
era_file = Path("../outputs/interpolated/ERA5_monthly_interpolated_variance.nc")
nclim_file = Path("../outputs/nclim/nclim_gridpoint_variance.nc")

# === Output file ===
output_dir = Path("../outputs/comparisons")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "ERA5_minus_NCLIM_monthly_variance.nc"

# === Load datasets ===
era_ds = xr.open_dataset(era_file)
nclim_ds = xr.open_dataset(nclim_file)

# === Mapping month names ===
months = ["jan","feb","mar","apr","may","jun",
          "jul","aug","sep","oct","nov","dec"]

# === Prepare an empty xarray Dataset to store differences ===
diff_ds = xr.Dataset(
    coords={
        "lat": era_ds.lat,
        "lon": era_ds.lon,
        "month": np.arange(1, 13)
    }
)

# === Compute difference ===
diff_data = []
for i, month in enumerate(months):
    era_var = f"var_{month}"     # <-- fixed name for ERA5 vars
    era_data = era_ds[era_var]
    nclim_data = nclim_ds["tavg_variance"].isel(month=i)
    
    diff = era_data - nclim_data
    diff_data.append(diff)

# Stack differences into a single variable
diff_ds["variance_diff"] = xr.concat(diff_data, dim="month")
diff_ds["variance_diff"].attrs["units"] = "degree_Celsius^2"
diff_ds["variance_diff"].attrs["description"] = "ERA5 minus NCLIM monthly variance"

# === Save to NetCDF ===
diff_ds.to_netcdf(output_file)
print(f"Difference file saved to {output_file}")
