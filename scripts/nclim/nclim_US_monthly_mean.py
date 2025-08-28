import xarray as xr
import numpy as np
import sys
from pathlib import Path

code_b_dir = Path("/global/homes/p/pchen704/code/era/new")
sys.path.append(str(code_b_dir))


from era_US_monthly_mean import compute_gridcell_area


ds = xr.open_dataset("nclim_monthly_means.nc")
tavg_da = ds["tavg_mean"]
lat = ds["lat"].values
lon = ds["lon"].values


area_da = compute_gridcell_area(lat, lon)
area_da = xr.DataArray(area_da, coords=[lat, lon], dims=["lat", "lon"])


weighted_sum = (tavg_da * area_da).sum(dim=["lat", "lon"])
total_area = area_da.sum(dim=["lat", "lon"])
weighted_avgs = (weighted_sum / total_area).values

ds_out = xr.Dataset(
    {"tavg_us": ("time", weighted_avgs)},
    coords={
        "year": ("time", ds["year"].values),
        "month": ("time", ds["month"].values),
        "time": np.arange(len(weighted_avgs))
    }
)

ds_out.to_netcdf("nclim_monthly_us_mean.nc")
print("Saved vectorized area-weighted U.S. monthly mean temperatures to nclim_monthly_us_mean.nc")
