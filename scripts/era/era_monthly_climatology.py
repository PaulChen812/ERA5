import xarray as xr
import numpy as np


infile = "ERA5_monthly_means_gridpoint.nc"
outfile = "ERA5_monthly_climatology_gridpoint.nc"

ds = xr.open_dataset(infile)
da = ds["t2m"]

month_names = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]

clim_vars = {}


months = da.time.dt.month

for i, m in enumerate(range(1, 13)):
    da_month = da.sel(time=months == m)
    da_clim = da_month.mean(dim="time") 
    var_name = f"t2m_{month_names[i]}"
    clim_vars[var_name] = da_clim


clim_ds = xr.Dataset(clim_vars)

clim_ds.to_netcdf(outfile)
print(f"Saved monthly climatology to {outfile}")
