import xarray as xr
import numpy as np


ds_means = xr.open_dataset("nclim_monthly_means.nc")
tavg_da = ds_means["tavg_mean"]
years = ds_means["year"].values
months = ds_means["month"].values
lat = ds_means["lat"].values
lon = ds_means["lon"].values

ds_clim = xr.open_dataset("nclim_monthly_climatology.nc")
climatology = ds_clim["tavg_climatology"]


monthly_variance = np.zeros((12, len(lat), len(lon)), dtype=np.float32)


for m in range(1, 13):
    month_mask = months == m

    month_data = tavg_da.values[month_mask] 
    

    diffs = month_data - climatology[m-1].values
    var = (diffs**2).mean(axis=0)  
    
    monthly_variance[m-1] = var


ds_out = xr.Dataset(
    {
        "tavg_variance": (("month", "lat", "lon"), monthly_variance, {"units": "degree_Celsius^2"})
    },
    coords={
        "month": np.arange(1, 13),
        "lat": lat,
        "lon": lon
    }
)

ds_out.to_netcdf("nclim_monthly_variance.nc")
print("Saved monthly variance fields to nclim_monthly_variance.nc")
