import os
import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm
from matplotlib import cm

class GlobalTemperaturePlotter:
    def __init__(self,inputs):
        if isinstance(inputs,str):
            if os.path.isdir(inputs):
                pattern = os.path.join(inputs,"*.nc")
                self.filenames = sorted(glob.glob(pattern))
            else:
                self.filenames = [inputs]
        elif isinstance(inputs, list):
            self.filename = inputs

    def __call__(self):
        ds = xr.open_mfdataset(
                self.filenames,
                combine="by_coords",
                parallel=True,
                chunks={"time": 8},
                engine="netcdf4",
                preprocess=self._preprocess_single
        )
        
        t2m = ds["t2m"] - 273.15
        t2m = t2m.where(~np.isnan(t2m))

        if "time" in t2m.dims:
            spatial_avg = t2m.mean(dim="time")
        else:
            spatial_avg = t2m


        lat = spatial_avg["latitude"]
        lon = spatial_avg["longitude"]
        print(lat.min().values, lat.max().values)
        
        cos_lat = np.cos(np.deg2rad(lat.values))
        cos_lat_clipped = np.maximum(cos_lat, 1e-10)
        weight_array = cos_lat_clipped[:,None] * np.ones((1,len(lon)))
        weights = xr.DataArray(
                weight_array,
                coords = [lat,lon],
                dims = ["latitude", "longitude"]
            )
        weighted_global_mean = (spatial_avg * weights).sum(dim=["latitude","longitude"]) / weights.sum()

        vmin = float(spatial_avg.min().compute())
        vmax = float(spatial_avg.max().compute())

        if vmin == vmax:
            vmin-= 0.5
            vmax+=0.5
        n_bins = 12
        levels = np.linspace(vmin, vmax, n_bins + 1)
        cmap = plt.get_cmap("coolwarm", n_bins)
        norm = BoundaryNorm(boundaries=levels, ncolors=cmap.N,clip = True)

        spatial_plot = spatial_avg.compute()
        wg_mean = weighted_global_mean.compute()
        

        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.Stereographic(central_longitude=0, central_latitude=90))
        # ax.set_global()
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", alpha=0.5)
        gl.top_labels= False
        gl.right_labels = False
        ax.set_extent([-180, 180, 30, 90], ccrs.PlateCarree())
        # print(lat.min().values, lat.max().values)

        mesh = spatial_plot.plot.pcolormesh(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
                add_colorbar=False
            )
        cbar = plt.colorbar(
                cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
                orientation="horizontal",
                pad=0.05,
                ticks=(levels[:-1] + levels[1:]) / 2,
                boundaries = levels,
                shrink=0.8
            )
        cbar.set_label("Avg Temp (C)")
        cbar.set_ticklabels([f"{x:.1f}" for x in (levels[:-1] + levels[1:]) / 2])

        title = (f"Global Mean Temp Over Jan 2025")
        plt.title(title)

        out_file = f"global_temp_01_2025.png"
        plt.savefig(out_file,dpi=300,bbox_inches="tight")
        plt.close()
    

    
    @staticmethod
    def _preprocess_single(ds):
        return ds

if __name__ == "__main__":
    plotter = GlobalTemperaturePlotter("/global/cfs/cdirs/m3638/ERA5_Data/2m_temperature_daily/2025/01")
    plotter()
