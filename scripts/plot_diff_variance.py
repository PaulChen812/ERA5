import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path

# === Input difference file ===
infile = Path("../outputs/comparisons/ERA5_minus_NCLIM_monthly_variance.nc")
ds = xr.open_dataset(infile)

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

lat = ds["lat"].values
lon = ds["lon"].values

# === Output directory for plots ===
output_dir = Path("../outputs/comparisons/variance")
output_dir.mkdir(parents=True, exist_ok=True)

# === Compute global min/max and round to nice numbers ===
data = ds["variance_diff"]
dmin = float(data.min())
dmax = float(data.max())

# Round to nearest 0.5
vmin = np.floor(dmin * 2) / 2
vmax = np.ceil(dmax * 2) / 2

print(f"Global variance diff range: {dmin:.3f} to {dmax:.3f}, using vmin={vmin}, vmax={vmax}")

# === Clip values to plotting range ===
data_clipped = data.clip(min=vmin, max=vmax)

# === Loop over months and plot ===
for i, month_name in enumerate(month_names):
    var_diff = data_clipped.isel(month=i).values

    plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.LambertConformal())
    ax.set_extent([-125, -66.5, 24.5, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))

    # Contour levels every 0.5
    levels = np.arange(vmin, vmax + 0.5, 0.5)

    mesh = ax.contourf(
        lon, lat, var_diff,
        levels=10,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        # extend='both'
    )

    cbar = plt.colorbar(mesh, orientation="vertical", pad=0.02)
    cbar.set_label("Variance Difference (°C²)")

    plt.title(f"ERA5 minus NCLIM Monthly Variance: {month_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"variance_diff_{month_name}.png", dpi=300)
    plt.close()
    print(f"Saved variance difference plot for {month_name}")

ds.close()
print("All monthly variance difference plots completed.")
