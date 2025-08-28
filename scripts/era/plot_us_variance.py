import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path



infile = Path("../../outputs/era/ERA5_monthly_variance.nc")
ds = xr.open_dataset(infile)


months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

lat = ds["latitude"].values
lon = ds["longitude"].values


output_dir = Path("../../outputs/era/variance")
output_dir.mkdir(parents=True, exist_ok=True)

for i, month in enumerate(months):

    var_name = f"var_{month}"
    if var_name not in ds.variables:
        raise KeyError(f"Variable {var_name} not found in file.")
    
    var_data = ds[var_name].values


    plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.LambertConformal())
    ax.set_extent([-125, -66.5, 24.5, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))

    mesh = ax.contourf(
        lon, lat, var_data,
        20, 
        transform=ccrs.PlateCarree(),
        cmap="viridis"
    )


    cbar = plt.colorbar(mesh, orientation="vertical", pad=0.02)
    cbar.set_label("Variance (°C²)")

    plt.title(f"ERA5 U.S. Monthly Variance: {month_names[i]}")
    plt.tight_layout()
    plt.savefig(output_dir / f"variance_{month_names[i]}.png", dpi=300)
    plt.close()
    print(f"Saved variance plot for {month_names[i]}")

ds.close()
print("All monthly variance plots completed.")
