import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path

# === Input dataset ===
infile = Path("../../outputs/nclim/nclim_gridpoint_variance.nc")
ds = xr.open_dataset(infile)

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

lat = ds["lat"].values
lon = ds["lon"].values

# === Output directory ===
output_dir = Path("../../outputs/nclim")
output_dir.mkdir(parents=True, exist_ok=True)

# === Compute global min/max across all months ===
dmin = float(ds["tavg_variance"].min())
dmax = float(ds["tavg_variance"].max())

# Round outward to nearest 0.5
vmin = np.floor(dmin * 2) / 2
vmax = np.ceil(dmax * 2) / 2

print(f"Global variance range: {dmin:.2f} to {dmax:.2f}, using vmin={vmin}, vmax={vmax}")

# === Loop over months and plot ===
for i, month_name in enumerate(month_names):
    var_temp = ds["tavg_variance"].isel(month=i).values

    plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.LambertConformal())
    ax.set_extent([-125, -66.5, 24.5, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))

    # Levels every 0.5
    levels = np.arange(vmin, vmax + 0.5, 0.5)

    mesh = ax.contourf(
        lon, lat, var_temp,
        levels=levels,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        extend="both"
    )

    cbar = plt.colorbar(mesh, orientation="vertical", pad=0.02)
    cbar.set_label("Temperature Variance (°C²)")

    plt.title(f"NCLIM U.S. Monthly Temperature Variance: {month_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"variance_{month_name}.png", dpi=300)
    plt.close()
    print(f"Saved variance plot for {month_name}")

ds.close()
print("All monthly variance plots completed.")
