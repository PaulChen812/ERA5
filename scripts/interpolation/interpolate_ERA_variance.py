import xarray as xr
import xesmf as xe
from pathlib import Path

# === Input files ===
era5_file = Path("../../outputs/era/ERA5_monthly_variance.nc")
nclim_file = Path("../../outputs/nclim/nclim_gridpoint_climatology.nc")

# === Output file ===
outfile = Path("../../outputs/era/ERA5_monthly_interpolated_variance.nc")

# Load datasets
era5_var = xr.open_dataset(era5_file)
nclim = xr.open_dataset(nclim_file)

# Define source and target grids
ds_in = xr.Dataset(
    {"lat": (["lat"], era5_var.latitude.values),
     "lon": (["lon"], era5_var.longitude.values)}
)

ds_out = xr.Dataset(
    {"lat": (["lat"], nclim.lat.values),
     "lon": (["lon"], nclim.lon.values)}
)

# Create bilinear regridder (weights reused if exist)
regridder = xe.Regridder(
    ds_in, ds_out, method="bilinear",
    filename="era5_to_nclim_bilinear_weights.nc",
    reuse_weights=True
)

# Interpolate each variance field
regridded_vars = {}
for var in era5_var.data_vars:
    print(f"Interpolating {var} ...")
    regridded_vars[var] = regridder(era5_var[var])

# Assemble new dataset on nClim grid
era5_var_on_nclim = xr.Dataset(regridded_vars, coords={"lat": nclim.lat, "lon": nclim.lon})

# Mask ERA5 variances where nClim has no data
mask = xr.where(nclim.tavg_climatology.isnull(), True, False).any(dim="month")
for var in era5_var_on_nclim.data_vars:
    era5_var_on_nclim[var] = era5_var_on_nclim[var].where(~mask)

# Save result
era5_var_on_nclim.to_netcdf(outfile)
print(f"Saved interpolated ERA5 variances to {outfile}")
