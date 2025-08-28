import xarray as xr
import numpy as np
from monthly_mean_weighting import create_us_mask, compute_gridcell_area, EARTH_RADIUS


climatology_file = "ERA5_monthly_climatology_gridpoint.nc"
variance_file = "ERA5_monthly_variance.nc"


outfile = "ERA5_US_monthly_clim_var.nc"


clim_ds = xr.open_dataset(climatology_file)
var_ds = xr.open_dataset(variance_file)


lat_vals = clim_ds.latitude.values
lon_vals = clim_ds.longitude.values

area_da = xr.DataArray(compute_gridcell_area(lat_vals, lon_vals), 
                       coords=[lat_vals, lon_vals], dims=["latitude", "longitude"])
us_mask_da = xr.DataArray(create_us_mask(lat_vals, lon_vals), 
                          coords=[lat_vals, lon_vals], dims=["latitude", "longitude"])

us_clim_vars = {}
us_var_vars = {}


month_names = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]


for m in month_names:

    clim_month = clim_ds[f"t2m_{m}"]
    clim_masked = clim_month.where(us_mask_da)
    area_masked = area_da.where(us_mask_da)
    
    total_area = area_masked.sum()
    weighted_sum = (clim_masked * area_masked).sum()
    us_clim_vars[f"US_mean_{m}"] = (weighted_sum / total_area).item()
    

    var_month = var_ds[f"var_{m}"]
    var_masked = var_month.where(us_mask_da)
    weighted_var_sum = (var_masked * area_masked).sum()
    us_var_vars[f"US_var_{m}"] = (weighted_var_sum / total_area).item()


out_ds = xr.Dataset({**us_clim_vars, **us_var_vars})


out_ds.to_netcdf(outfile)
print(f"Saved area-weighted U.S. climatology and variance to {outfile}")
