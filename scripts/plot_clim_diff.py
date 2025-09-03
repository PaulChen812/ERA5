import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path

# === Load the difference data (already regridded) ===
infile = Path("../outputs/nclim_vs_era5_climatology_diff.nc")
diff = xr.open_dataarray(infile)

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

lat = diff["latitude"].values
lon = diff["longitude"].values

output_dir = Path("../outputs")
output_dir.mkdir(parents=True, exist_ok=True)

for i, month in enumerate(months, start=1):
    # Select monthly slice
    temp_diff = diff.sel(month=i).values

    plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.LambertConformal())
    ax.set_extent([-125, -66.5, 24.5, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))

    # Plot differences
    mesh = ax.contourf(
        lon, lat, temp_diff,
        20,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=-5, vmax=5    # keep color range consistent
    )

    cbar = plt.colorbar(mesh, orientation="vertical", pad=0.02)
    cbar.set_label("Climatology Difference (nClimGrid – ERA5, °C)")

    plt.title(f"Monthly Climatology Difference: {month}")
    plt.tight_layout()
    plt.savefig(output_dir / f"climatology_diff_{month}.png", dpi=300)
    plt.close()
    print(f"Saved difference plot for {month}")

print("All monthly difference plots completed.")
