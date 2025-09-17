import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# === Input files ===
clim_file = Path("../../outputs/nclim/nclim_gridpoint_climatology.nc")
var_file  = Path("../../outputs/nclim/nclim_gridpoint_variance.nc")

# === Load datasets ===
ds_clim = xr.open_dataset(clim_file)
ds_var  = xr.open_dataset(var_file)

# Extract variables
clim = ds_clim["tavg_climatology"]  # (month, lat, lon)
var  = ds_var["tavg_variance"]      # (month, lat, lon)
lat  = ds_clim["lat"]

# === Compute area weights ===
# Weight ~ cos(lat) normalized to sum=1
weights = np.cos(np.deg2rad(lat))
weights = weights / weights.sum()

# Expand weights to match (month, lat, lon)
weights_2d = np.cos(np.deg2rad(lat))
weights_2d = weights_2d / weights_2d.sum()
weights_2d = xr.DataArray(weights_2d, dims=["lat"], coords={"lat": lat})

# Apply weights to climatology & variance
clim_mean = (clim.weighted(weights_2d)).mean(dim=("lat", "lon"))
var_mean  = (var.weighted(weights_2d)).mean(dim=("lat", "lon"))

# === Create output dataset ===
out_ds = xr.Dataset(
    {
        "NCLIM_mean": (("month",), clim_mean.values, {"units": "degC"}),
        "NCLIM_var":  (("month",), var_mean.values, {"units": "degC^2"}),
    },
    coords={"month": np.arange(1, 13)}
)

# Save to netCDF
out_file = Path("../../outputs/nclim/nclim_US_monthly_clim_var.nc")
out_ds.to_netcdf(out_file)

print(f"Saved area-weighted monthly climatology + variance to {out_file}")


