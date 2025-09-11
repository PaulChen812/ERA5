import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import numpy as np

# === Input file ===
infile = Path("../../outputs/interpolated/ERA5_monthly_climatology_on_nclim_grid.nc")
ds = xr.open_dataset(infile)

months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

lat = ds["lat"].values
lon = ds["lon"].values

# === Output directory ===
output_dir = Path("../../outputs/interpolated/climatology")
output_dir.mkdir(parents=True, exist_ok=True)

# === Compute global min/max across all months ===
all_data = xr.concat([ds[f"t2m_{m}"] for m in months], dim="month")
dmin = float(all_data.min())
dmax = float(all_data.max())

# Round outward to nearest 0.5 for “nice” colorbar
vmin = np.floor(dmin * 2) / 2
vmax = np.ceil(dmax * 2) / 2

print(f"Global climatology range: {dmin:.2f} to {dmax:.2f}, using vmin={vmin}, vmax={vmax}")

# === Loop over months and plot ===
for i, month in enumerate(months):
    var_name = f"t2m_{month}"
    temp = ds[var_name].values

    plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.LambertConformal())
    ax.set_extent([-125, -66.5, 24.5, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))

    # Use consistent levels for all months
    levels = np.arange(vmin, vmax + 0.5, 0.5)

    mesh = ax.contourf(
        lon, lat, temp,
        levels=levels,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        extend="both"
    )

    cbar = plt.colorbar(mesh, orientation="vertical", pad=0.02)
    cbar.set_label("Temperature (°C)")

    plt.title(f"ERA5 on nClim Grid – U.S. Monthly Climatology: {month_names[i]}")
    plt.tight_layout()
    plt.savefig(output_dir / f"interpolated_{month_names[i]}.png", dpi=300)
    plt.close()
    print(f"Saved plot for {month_names[i]}")

ds.close()
print("All monthly plots completed.")
