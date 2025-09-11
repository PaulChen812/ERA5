import xarray as xr
import xesmf as xe
from pathlib import Path

# === Input files ===
era5_file = Path("../outputs/era/ERA5_monthly_climatology_gridpoint.nc")
nclim_file = Path("../outputs/nclim/nclim_gridpoint_climatology.nc")
weights_file = "era5_to_nclim_bilinear_weights.nc"
# === Output file ===
outfile = Path("../outputs/era/ERA5_monthly_climatology_on_nclim_grid.nc")

# --- Load datasets ---
era5 = xr.open_dataset(era5_file)
nclim = xr.open_dataset(nclim_file)

# Create source and target grids
ds_in = xr.Dataset(
    {
        "lat": (["lat"], era5.latitude.values),
        "lon": (["lon"], era5.longitude.values),
    }
)

ds_out = xr.Dataset(
    {
        "lat": (["lat"], nclim.lat.values),
        "lon": (["lon"], nclim.lon.values),
    }
)

if Path(weights_file).exists():
# Create regridder (bilinear interpolation)
    regridder = xe.Regridder(ds_in, ds_out, method="bilinear", 
                            filename="era5_to_nclim_bilinear_weights.nc", 
                            reuse_weights=True)

else:
    regridder = xe.Regridder(
        ds_in, ds_out, method="bilinear",
        filename = "era5_to_nclim_bilinear_weights.nc",
        reuse_weights=False
    )

# Regrid each ERA5 monthly climatology field
regridded_vars = {}
for var in era5.data_vars:
    print(f"Regridding {var} ...")
    regridded_vars[var] = regridder(era5[var])

# Combine into a new dataset
era5_on_nclim = xr.Dataset(regridded_vars, coords={"lat": nclim.lat, "lon": nclim.lon})

# Mask ERA5 data where nClimGrid has no data
mask = xr.where(nclim.tavg_climatology.isnull(), True, False).any(dim="month")
for var in era5_on_nclim.data_vars:
    era5_on_nclim[var] = era5_on_nclim[var].where(~mask)

# Save to file
era5_on_nclim.to_netcdf(outfile)
print(f"Saved regridded ERA5 climatology to {outfile}")
