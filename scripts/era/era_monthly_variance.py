import xarray as xr
import numpy as np

monthly_means_file = "ERA5_monthly_means_gridpoint.nc"
climatology_file = "ERA5_monthly_climatology_gridpoint.nc" 
outfile = "ERA5_monthly_variance.nc"

ds_means = xr.open_dataset(monthly_means_file)
ds_clim = xr.open_dataset(climatology_file)

da_means = ds_means["t2m"]
months = da_means.time.dt.month

month_names = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
variance_vars = {}

for i, m in enumerate(range(1, 13)):
    da_month = da_means.sel(time=months == m)
    
    clim = ds_clim[f"t2m_{month_names[i]}"]
    
    sq_diff = (da_month - clim) ** 2
    
    var_month = sq_diff.mean(dim="time")
    
    var_name = f"var_{month_names[i]}"
    variance_vars[var_name] = var_month

var_ds = xr.Dataset(variance_vars)
var_ds.to_netcdf(outfile)
print(f"Saved monthly variance per gridpoint to {outfile}")
