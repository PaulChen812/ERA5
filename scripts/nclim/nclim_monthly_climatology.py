import xarray as xr
import numpy as np


ds = xr.open_dataset("nclim_monthly_means.nc")
tavg_da = ds["tavg_mean"]
years = ds["year"].values
months = ds["month"].values
lat = ds["lat"].values
lon = ds["lon"].values


climatology = np.zeros((12, len(lat), len(lon)), dtype=np.float32)


for m in range(1, 13):
    month_mask = months == m
    climatology[m-1] = tavg_da.values[month_mask].mean(axis=0)


ds_out = xr.Dataset(
    {
        "tavg_climatology": (("month", "lat", "lon"), climatology, {"units": "degree_Celsius"})
    },
    coords={
        "month": np.arange(1, 13),
        "lat": lat,
        "lon": lon
    }
)

ds_out.to_netcdf("nclim_gridpoint_climatology.nc")
print("Saved monthly climatology fields to nclim_gridpoint_climatology.nc")
