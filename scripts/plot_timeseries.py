import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# === File paths ===
era5_interp_clim = Path("../outputs/interpolated/ERA5_area_weighted_climatology.nc")
era5_interp_var  = Path("../outputs/interpolated/ERA5_area_weighted_variance.nc")
nclim_file       = Path("../outputs/nclim/nclim_US_monthly_clim_var.nc")
era5_us_file     = Path("../outputs/era/ERA5_US_monthly_clim_var.nc")

# === Month labels ===
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# === Load datasets ===
ds_interp_clim = xr.open_dataset(era5_interp_clim)
ds_interp_var  = xr.open_dataset(era5_interp_var)
ds_nclim       = xr.open_dataset(nclim_file)
ds_era5_us     = xr.open_dataset(era5_us_file)

# --- Extract interpolated ERA5 (already stacked by month) ---
interp_clim = ds_interp_clim["t2m"].values
interp_var  = ds_interp_var["variance"].values

# --- Extract nClimGrid ---
nclim_clim = ds_nclim["NCLIM_mean"].values
nclim_var  = ds_nclim["NCLIM_var"].values

# --- Extract original ERA5 US data ---
era5_clim = np.array([ds_era5_us[f"US_mean_{m.lower()}"].values for m in months])
era5_var  = np.array([ds_era5_us[f"US_var_{m.lower()}"].values  for m in months])

# === Plot Climatology ===
plt.figure(figsize=(10,6))
plt.plot(months, era5_clim, marker="o", label="ERA5 US (Original)")
plt.plot(months, interp_clim, marker="s", label="ERA5 Interpolated (Global → US grid)")
plt.plot(months, nclim_clim, marker="^", label="nClimGrid US")
plt.title("Monthly Climatology")
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("../outputs/monthly_climatology_comparison.png", dpi=300)
plt.close()

# === Plot Variance ===
plt.figure(figsize=(10,6))
plt.plot(months, era5_var, marker="o", label="ERA5 US (Original)")
plt.plot(months, interp_var, marker="s", label="ERA5 Interpolated (Global → US grid)")
plt.plot(months, nclim_var, marker="^", label="nClimGrid US")
plt.title("Monthly Variance")
plt.xlabel("Month")
plt.ylabel("Variance (°C²)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("../outputs/monthly_variance_comparison.png", dpi=300)
plt.close()

print("Saved plots to outputs/monthly_climatology_comparison.png and monthly_variance_comparison.png")
