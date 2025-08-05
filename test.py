import os
import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import dask
from dask.diagnostics import ProgressBar

class GlobalTemperaturePlotter:
    def __init__(self, inputs):
        if isinstance(inputs, str):
            if os.path.isdir(inputs):
                pattern = os.path.join(inputs, "*.nc")
                self.filenames = sorted(glob.glob(pattern))
            else:
                self.filenames = [inputs]
        elif isinstance(inputs, list):
            self.filenames = inputs
        else:
            raise ValueError("Inputs must be a directory or list of file paths.")

        if not self.filenames:
            raise RuntimeError("No NetCDF files found.")

    def __call__(self):
        # Lazy open all files together
        ds = xr.open_mfdataset(
            self.filenames,
            combine="by_coords",
            parallel=True,
            chunks={"time": 8},
            engine="netcdf4",
            preprocess=self._preprocess_single  # ensures consistent decoding
        )

        # Extract temperature and convert to Celsius (ERA5 t2m is in Kelvin)
        t2m = ds["t2m"] - 273.15

        # Mask potential fill values / NaNs (xarray usually does this via decode_cf)
        t2m = t2m.where(~np.isnan(t2m))

        # Compute the average over time (if present) to yield a spatial map
        if "time" in t2m.dims:
            spatial_avg = t2m.mean(dim="time")
        else:
            spatial_avg = t2m

        # Diagnostics: global percentiles to help spot if map is plausible
        with dask.diagnostics.ProgressBar():
            p10, p50, p90 = dask.compute(
                spatial_avg.quantile(0.1),
                spatial_avg.quantile(0.5),
                spatial_avg.quantile(0.9),
            )

        print(f"Spatial average percentiles: 10th={float(p10.values):.2f}°C, "
              f"median={float(p50.values):.2f}°C, 90th={float(p90.values):.2f}°C")

        # Compute area weights (cosine latitude with floor)
        lat = spatial_avg["latitude"]
        lon = spatial_avg["longitude"]
        # cos(lat) weighting; floor small values to avoid zero at poles
        weight_array = np.maximum(np.cos(np.deg2rad(lat.values))[:, None], 1e-10)
        weights = xr.DataArray(
            weight_array,
            coords=[lat, lon],
            dims=["latitude", "longitude"]
        )

        # Weighted global mean scalar
        weighted_global_mean = (spatial_avg * weights).sum(dim=["latitude", "longitude"]) / weights.sum()

        # Determine discrete colormap bins from the spatial map
        vmin = float(spatial_avg.min().compute())
        vmax = float(spatial_avg.max().compute())
        if vmin == vmax:  # degenerate
            vmin -= 0.5
            vmax += 0.5
        n_bins = 12
        levels = np.linspace(vmin, vmax, n_bins + 1)
        cmap = cm.get_cmap("coolwarm", n_bins)
        norm = BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=True)

        # Compute the map now (trigger computation)
        with ProgressBar():
            spatial_plot = spatial_avg.compute()
            wg_mean = weighted_global_mean.compute()

        # Plot
        fig = plt.figure(figsize=(12, 8))
        # Choose stereographic centered at lat=30N for a more balanced global look (adjust if desired)
        ax = plt.axes(projection=ccrs.Stereographic(central_longitude=0, central_latitude=30))
        ax.set_global()
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", alpha=0.6)
        gl.top_labels = False
        gl.right_labels = False

        mesh = spatial_plot.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            add_colorbar=False
        )

        # Discrete colorbar with bin centers
        cbar = plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            orientation="horizontal",
            pad=0.05,
            ticks=(levels[:-1] + levels[1:]) / 2,
            boundaries=levels,
            shrink=0.8
        )
        cbar.set_label("Avg Temp (°C)")
        cbar.set_ticklabels([f"{x:.1f}" for x in (levels[:-1] + levels[1:]) / 2])

        title = (f"Aggregated Global Mean 2m Temp over {len(self.filenames)} file(s) — "
                 f"Weighted Mean: {wg_mean.values:.2f} °C")
        plt.title(title)

        out_file = f"global_temp_aggregated_{len(self.filenames)}.png"
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved aggregated map to {out_file}")

    @staticmethod
    def _preprocess_single(ds):
        # ensure t2m exists and decode properly; placeholder if further harmonization needed
        return ds

if __name__ == "__main__":
    raw = input("Enter directory or comma-separated list of .nc files: ").strip()
    if "," in raw:
        files = [p.strip() for p in raw.split(",") if p.strip()]
        plotter = GlobalTemperaturePlotter(files)
    else:
        plotter = GlobalTemperaturePlotter(raw)
    plotter()

