#!/usr/bin/env python3
import xarray as xr
import regionmask
import cartopy.io.shapereader as shpreader
from pathlib import Path

def create_us_mask(lat, lon):
    """
    Create a boolean mask for grid points inside the U.S.
    """
    shpfilename = shpreader.natural_earth(
        resolution='50m',
        category='cultural',
        name='admin_0_countries'
    )
    reader = shpreader.Reader(shpfilename)
    countries = list(reader.records())

    # Find USA polygon(s)
    us_poly = [c.geometry for c in countries
               if c.attributes['ADMIN'] == 'United States of America']

    # Make regionmask object
    mask = regionmask.Regions([us_poly[0]])
    mask_2d = mask.mask(lon, lat)  # shape (lat, lon)

    # Return boolean: True inside US
    return (mask_2d == 0)

def main():
    input_file = Path("ERA_Monthly_mean.nc")
    output_file = Path("ERA_US_Monthly_mean.nc")

    if not input_file.exists():
        raise FileNotFoundError(f"{input_file} not found.")

    print(f"Opening {input_file}...")
    ds = xr.open_dataset(input_file)

    # Assume variable name is unknown; take first data variable
    var_name = list(ds.data_vars)[0]
    print(f"Variable found: {var_name}")

    # Create US mask matching dataset grid
    print("Creating US mask...")
    us_mask = create_us_mask(ds.latitude.values, ds.longitude.values)

    # Apply mask to data
    print("Applying mask and computing US monthly mean...")
    masked_data = ds[var_name].where(us_mask)

    # Compute area-weighted mean for US
    import numpy as np
    R = 6371000  # Earth radius in m
    lat_rad = np.deg2rad(ds.latitude)
    dlat = np.abs(np.diff(lat_rad).mean())
    dlon = np.deg2rad(np.abs(np.diff(ds.longitude).mean()))
    cell_areas = (R**2 * dlat * dlon *
                  np.cos(lat_rad)[:, np.newaxis])

    # Convert to DataArray
    area_da = xr.DataArray(cell_areas,
                           coords=[ds.latitude, ds.longitude],
                           dims=["latitude", "longitude"])

    # Weighted mean
    us_mean_series = (masked_data * area_da).sum(dim=("latitude", "longitude")) / \
                     area_da.where(~masked_data.isnull()).sum(dim=("latitude", "longitude"))

    # Save result
    print(f"Saving US monthly mean to {output_file}...")
    us_mean_series.to_netcdf(output_file)

    print("Done!")

if __name__ == "__main__":
    main()
