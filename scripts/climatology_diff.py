import xarray as xr
from pathlib import Path
import numpy as np

# === Input files ===
era_file = Path("../outputs/interpolated/ERA5_monthly_climatology_on_nclim_grid.nc")
nclim_file = Path("../outputs/nclim/nclim_gridpoint_climatology.nc")

# === Output file ===
output_dir = Path("../outputs/comparisons")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "ERA5_minus_NCLIM_monthly_climatology.nc"

# === Load datasets ===
era_ds = xr.open_dataset(era_file)
nclim_ds = xr.open_dataset(nclim_file)

# === Mapping month names ===
months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]

# === Prepare an empty xarray Dataset to store differences ===
diff_ds = xr.Dataset(
    coords={
        "lat": era_ds.lat,
        "lon": era_ds.lon,
        "month": np.arange(1,13)
    }
)

# === Compute difference ===
diff_data = []
for i, month in enumerate(months):
    era_var = f"t2m_{month}"
    
    # Ensure ERA and NCLIM have the same shape
    era_data = era_ds[era_var]
    nclim_data = nclim_ds["tavg_climatology"].isel(month=i)
    
    # Compute difference
    diff = era_data - nclim_data
    diff_data.append(diff)

# Stack differences into a single variable
diff_ds["t2m_diff"] = xr.concat(diff_data, dim="month")
diff_ds["t2m_diff"].attrs["units"] = "degree_Celsius"
diff_ds["t2m_diff"].attrs["description"] = "ERA5 minus NCLIM monthly climatology"

# === Save to NetCDF ===
diff_ds.to_netcdf(output_file)
print(f"Difference file saved to {output_file}")
