import os
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import matplotlib.path as mpath
from cartopy.util import add_cyclic_point

MELTING_TEMP = 273.15

class GlobalTemperatureAnalyzer:
    def __init__(self, directory: str | Path, latitude_range: tuple[float, float] = (90, -90), variable: str = "t2m"):
        if not isinstance(directory, (str, Path)):
            raise ValueError("directory must be a str or Path")
        self.directory = Path(directory)
        self.latitude_range = latitude_range
        self.variable = variable


        parts = self.directory.parts[-2:]
        try:
            self.year = int(parts[0])
            self.month = int(parts[1])
        except Exception:
            raise ValueError("Directory path must end with /YYYY/MM (e.g. /2025/01)")

    def preprocess(self, filename: Path) -> xr.DataArray:
        ds = xr.open_dataset(filename)
        if self.variable not in ds:
            raise RuntimeError(f"Variable '{self.variable}' not found in dataset.")
        var_sel = ds[self.variable].sel(latitude=slice(*self.latitude_range)) - MELTING_TEMP
        return var_sel.mean(dim="time")

    def plot_2d(self, spatial_avg: xr.DataArray, title: str, out_file: str):

        spatial_avg_cyclic, lon_cyclic = add_cyclic_point(
            spatial_avg.values,
            coord=spatial_avg["longitude"].values,
            axis=1
        )
        lat = spatial_avg["latitude"].values

        vmin = float(np.nanmin(spatial_avg_cyclic))
        vmax = float(np.nanmax(spatial_avg_cyclic))
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5

        n_bins = 12
        levels = np.linspace(vmin, vmax, n_bins + 1)
        cmap = plt.get_cmap("coolwarm", n_bins)
        norm = BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=True)

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=ccrs.Stereographic(central_longitude=0, central_latitude=90))
        ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False


        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

  
        cf = ax.contourf(
            lon_cyclic,
            lat,
            spatial_avg_cyclic,
            transform=ccrs.PlateCarree(),
            levels=levels,
            cmap=cmap,
            norm=norm
        )

        cbar = plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            orientation="horizontal",
            pad=0.05,
            ticks=(levels[:-1] + levels[1:]) / 2,
            shrink=0.8
        )
        cbar.set_label("Avg Temp (°C)")
        cbar.set_ticklabels([f"{x:.1f}" for x in (levels[:-1] + levels[1:]) / 2])
        plt.title(title)

        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()

    def __call__(self):

        files = sorted(self.directory.glob("*.nc"))
        if len(files) == 0:
            raise RuntimeError(f"No .nc files found in {self.directory}")

        total = None
        for file in files:
            print(f"Processing {file.name} ...")
            daily_mean = self.preprocess(file)
            total = daily_mean if total is None else total + daily_mean

        monthly_mean = total / len(files)
        print(f"Monthly mean computed for {self.year}-{self.month:02d}")

        title = f"Global Mean Temp for {self.year}-{self.month:02d}"
        out_file = f"global_temp_{self.year}_{self.month:02d}.png"
        self.plot_2d(monthly_mean, title=title, out_file=out_file)


if __name__ == "__main__":

    input_dir = "/global/cfs/cdirs/m3638/ERA5_Data/2m_temperature_daily/2025/01"
    gta = GlobalTemperatureAnalyzer(input_dir)
    gta()
