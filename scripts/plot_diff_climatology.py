import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path

# === Input difference file ===
infile = Path("../outputs/comparisons/ERA5_minus_NCLIM_monthly_climatology.nc")
ds = xr.open_dataset(infile)

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

lat = ds["lat"].values
lon = ds["lon"].values

# === Output directory for plots ===
output_dir = Path("../outputs/comparisons/climatology")
output_dir.mkdir(parents=True, exist_ok=True)

# === Clip values to -10 to +10°C for plotting ===
data_clipped = ds["t2m_diff"].clip(min=-5, max=5)



# === Loop over months and plot ===
for i, month_name in enumerate(month_names):
    temp_diff = data_clipped.isel(month=i).values

    plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.LambertConformal())
    ax.set_extent([-125, -66.5, 24.5, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))



    mesh = ax.contourf(
        lon, lat, temp_diff,
        levels=10,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        # extend='both'
    )

    cbar = plt.colorbar(mesh, orientation="vertical", pad=0.02, shrink = 0.55)
    cbar.set_label("Temperature Difference (°C)")

    plt.title(f"ERA5 minus NCLIM Monthly Climatology: {month_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"climatology_diff_{month_name}.png", dpi=300)
    plt.close()
    print(f"Saved difference plot for {month_name}")

ds.close()
print("All monthly difference plots completed.")
