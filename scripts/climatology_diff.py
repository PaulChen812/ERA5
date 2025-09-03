import xarray as xr
import numpy as np
import xesmf as xe
from pathlib import Path

# === Load datasets ===
nclim_file = Path("../outputs/nclim/nclim_gridpoint_climatology.nc")
era5_file  = Path("../outputs/era/ERA5_monthly_climatology_gridpoint.nc")
weights_file = "nclim_to_era5_weights.nc"

ds_nclim = xr.open_dataset(nclim_file)
ds_era5  = xr.open_dataset(era5_file)

# nClim variables: tavg_climatology(month, lat, lon)
# ERA5 variables: t2m_jan, t2m_feb, ... (monthly fields already expanded)

# --- Step 1: Put ERA5 into a consistent format (month, lat, lon) ---
era5_months = [
    "t2m_jan", "t2m_feb", "t2m_mar", "t2m_apr",
    "t2m_may", "t2m_jun", "t2m_jul", "t2m_aug",
    "t2m_sep", "t2m_oct", "t2m_nov", "t2m_dec"
]

era5_data = xr.concat([ds_era5[var] for var in era5_months], dim="month")
era5_data = era5_data.assign_coords(month=np.arange(1, 13))
era5_data.name = "tavg_climatology"

# --- Step 2: Build regridder from nClimGrid to ERA5 grid ---
if Path(weights_file).exists():
    regridder = xe.Regridder(
        ds_nclim,
        era5_data,
        method = "conservative",
        filename = weights_file,
        reuse_weights = True
    )
else: 
    regridder = xe.Regridder(
        ds_nclim,
        era5_data,
        method="conservative",   # area-weighted average
        filename=weights_file,
        reuse_weights=False
    )

# --- Step 3: Apply regridding (downscale nClimGrid to ERA5 resolution) ---
nclim_on_era5 = regridder(ds_nclim["tavg_climatology"])

# Now both datasets have same shape: (month, lat, lon)

# --- Step 4: Compute differences ---
diff = nclim_on_era5 - era5_data

# === Optional: Save to file for later use ===
out_file = Path("../outputs/nclim_vs_era5_climatology_diff.nc")
diff.to_netcdf(out_file)
