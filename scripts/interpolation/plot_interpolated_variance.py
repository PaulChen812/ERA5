import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import numpy as np

# === Input file ===
infile = Path("../../outputs/interpolated/ERA5_monthly_interpolated_variance.nc")
ds = xr.open_dataset(infile)

months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

lat = ds["lat"].values
lon = ds["lon"].values

# === Output directory ===
output_dir = Path("../../outputs/interpolated/variance")
output_dir.mkdir(parents=True, exist_ok=True)

# === Compute global min/max across all months ===
all_data = xr.concat([ds[f"var_{m}"] for m in months], dim="month")


# === Loop over months and plot ===
for i, month in enumerate(months):
    var_name = f"var_{month}"   # adjust if needed
    variance = ds[var_name].values

    fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw={"projection": ccrs.LambertConformal()}
    )

    ax.set_extent([-125, -66.5, 24.5, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))



    mesh = ax.contourf(
        lon, lat, variance,
        levels=20,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        # extend="both"
    )

    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=1.0, pad=0.02)
    cbar.set_label("Temperature Variance (°C²)")

    plt.title(f"ERA5 on nClim Grid – U.S. Monthly Variance: {month_names[i]}")
    plt.tight_layout()
    plt.savefig(output_dir / f"variance_{month_names[i]}.png", dpi=300)
    plt.close(fig)
    print(f"Saved variance plot for {month_names[i]}")

ds.close()
print("All monthly variance plots completed.")
