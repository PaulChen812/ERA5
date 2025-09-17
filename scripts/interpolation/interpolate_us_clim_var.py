import xarray as xr
import numpy as np
from pathlib import Path

# === Input files ===
clim_file = Path("../../outputs/interpolated/ERA5_monthly_climatology_on_nclim_grid.nc")
var_file  = Path("../../outputs/interpolated/ERA5_monthly_interpolated_variance.nc")

# === Output files ===
clim_outfile = Path("../../outputs/interpolated/ERA5_area_weighted_climatology.nc")
var_outfile  = Path("../../outputs/interpolated/ERA5_area_weighted_variance.nc")

# === Month names ===
months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]

# === Load datasets ===
clim = xr.open_dataset(clim_file)
var  = xr.open_dataset(var_file)

# --- Area weights (cos(lat)) ---
weights = np.cos(np.deg2rad(clim.lat))
weights = xr.DataArray(weights, coords=[clim.lat], dims=["lat"])

# === Process climatology ===
clim_means = []
for m in months:
    varname = f"t2m_{m}"
    print(f"Processing climatology {varname} ...")
    weighted_mean = clim[varname].weighted(weights).mean(dim=("lat", "lon"), skipna=True)
    clim_means.append(weighted_mean)

# Stack into one variable with month dimension
clim_out = xr.Dataset(
    {"t2m": xr.DataArray(clim_means, coords={"month": months}, dims=["month"])}
)
clim_out.to_netcdf(clim_outfile)
print(f"Saved area-weighted climatology to {clim_outfile}")

# === Process variance ===
var_means = []
for m in months:
    varname = f"var_{m}"
    print(f"Processing variance {varname} ...")
    weighted_mean = var[varname].weighted(weights).mean(dim=("lat", "lon"), skipna=True)
    var_means.append(weighted_mean)

var_out = xr.Dataset(
    {"variance": xr.DataArray(var_means, coords={"month": months}, dims=["month"])}
)
var_out.to_netcdf(var_outfile)
print(f"Saved area-weighted variance to {var_outfile}")
