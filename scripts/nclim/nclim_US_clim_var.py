import xarray as xr
import numpy as np
from pathlib import Path
import sys


area = Path("/global/homes/p/pchen704/code/era/new")
sys.path.append(str(area))
from era_US_monthly_mean import compute_gridcell_area


ds_clim = xr.open_dataset("nclim_monthly_climatology.nc")
clim_da = ds_clim["tavg_climatology"]

ds_var = xr.open_dataset("nclim_monthly_variance.nc")
var_da = ds_var["tavg_variance"]    

lat = ds_clim["lat"].values
lon = ds_clim["lon"].values


area_da = compute_gridcell_area(lat, lon)
area_da = xr.DataArray(area_da, coords=[lat, lon], dims=["lat", "lon"])


total_area = area_da.sum(dim=["lat", "lon"])

clim_weighted = (clim_da * area_da).sum(dim=["lat", "lon"]) / total_area
var_weighted = (var_da * area_da).sum(dim=["lat", "lon"]) / total_area


ds_out = xr.Dataset(
    {
        "tavg_climatology_us": ("month", clim_weighted.values),
        "tavg_variance_us": ("month", var_weighted.values)
    },
    coords={
        "month": np.arange(1, 13)
    }
)

ds_out.to_netcdf("nclim_monthly_us_climatology_variance.nc")
print("Saved area-weighted U.S. climatology and variance to nclim_monthly_us_climatology_variance.nc")
