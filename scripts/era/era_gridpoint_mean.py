import os
from pathlib import Path
import xarray as xr
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import regionmask

MT = 273.15 

class MonthlyGridMeanCalculator:
    def __init__(self, directory, variable, latitude_range=None, longitude_range=None):
        self.directory = Path(directory)
        self.variable = variable
        self.latitude_range = latitude_range
        self.longitude_range = longitude_range

        # Prepare US regionmask for contiguous states
        self.us_states = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
        self.contiguous_indices = [
            i for i, abbr in enumerate(self.us_states.abbrevs)
            if abbr not in ("AK", "HI", "PR")
        ]

    def _lat_slice(self, lat_values: np.ndarray):
        if self.latitude_range is None:
            return slice(None)
        lo, hi = self.latitude_range
        if lat_values[0] > lat_values[-1]:
            # descending (90 → -90), flip the slice
            return slice(hi, lo)
        else:
            # ascending (-90 → 90)
            return slice(lo, hi)

    def _lon_slice(self, lon_values: np.ndarray):
        if self.longitude_range is None:
            return slice(None)
        lo, hi = self.longitude_range
        if lon_values[0] > lon_values[-1]:
            # descending
            return slice(hi, lo)
        else:
            # ascending
            return slice(lo, hi)

    def _us_mask(self, lat_vals, lon_vals):
        # Convert longitude to -180 to 180 if needed
        lon_vals_fixed = np.where(lon_vals > 180, lon_vals - 360, lon_vals)
        lon2d, lat2d = np.meshgrid(lon_vals_fixed, lat_vals)
        mask = self.us_states.mask(lon2d, lat2d)
        mask_bool = np.isin(mask, self.contiguous_indices)
        return xr.DataArray(
            mask_bool,
            coords={"latitude": lat_vals, "longitude": lon_vals},
            dims=("latitude", "longitude"),
        )

    def preprocess(self, filename: Path) -> xr.DataArray:
        ds = xr.open_dataset(filename)
        if self.variable not in ds:
            ds.close()
            raise RuntimeError(f"Variable '{self.variable}' not found in dataset.")

        # --- Normalize longitudes to -180..180 ---
        if (ds["longitude"] > 180).any():
            ds = ds.assign_coords(
                longitude=(((ds["longitude"] + 180) % 360) - 180)
            ).sortby("longitude")

        # --- Apply latitude + longitude slices ---
        lat_slice = self._lat_slice(ds["latitude"].values)
        lon_slice = self._lon_slice(ds["longitude"].values)
        var_sel = ds[self.variable].sel(latitude=lat_slice, longitude=lon_slice) - MT

        lat_vals_sel = var_sel.latitude.values
        lon_vals_sel = var_sel.longitude.values

        # --- Apply US mask (may still mask everything out, but slicing alone will work) ---
        us_mask = self._us_mask(lat_vals_sel, lon_vals_sel)
        var_sel = var_sel.where(us_mask)

        if np.all(np.isnan(var_sel.values)):
            ds.close()
            raise ValueError(f"All grid points masked out for file {filename}")

        daily_mean = var_sel.mean(dim="time")
        ds.close()
        return daily_mean

    def monthly_mean(self, year: int, month: int) -> xr.DataArray:
        month_dir = self.directory / f"{year:04d}" / f"{month:02d}"
        files = sorted(month_dir.glob("*.nc"))
        if not files:
            raise FileNotFoundError(f"No .nc files found in {month_dir}")
        total = None
        for f in files:
            dm = self.preprocess(f)
            total = dm if total is None else total + dm
        monthly_mean_da = total / len(files)
        monthly_mean_da = monthly_mean_da.expand_dims(
            time=[np.datetime64(f"{year}-{month:02d}")]
        )
        return monthly_mean_da


if __name__ == "__main__":
    directory = "/global/cfs/cdirs/m3638/ERA5_Data/2m_temperature_daily/"
    calc = MonthlyGridMeanCalculator(
        directory=directory, variable="t2m", latitude_range=(25, 50), longitude_range=(-125, -66.5)
    )

    years = range(1951, 2025)
    months = range(1, 13)
    year_month_pairs = list(itertools.product(years, months))

    results = []

    max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    print(f"Using max_workers = {max_workers}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(calc.monthly_mean, y, m): (y, m) for y, m in year_month_pairs}
        for future in as_completed(futures):
            y, m = futures[future]
            try:
                monthly_field = future.result()
                results.append(monthly_field)
                print(f"Finished {y}-{m:02d}")
            except FileNotFoundError:
                print(f"No data for {y}-{m:02d}")
            except Exception as e:
                print(f"Error for {y}-{m:02d}: {e}")

    if results:
        ds_out = xr.concat(results, dim="time")
        ds_out = ds_out.sortby("time")
        ds_out.to_netcdf("ERA5_monthly_means_gridpoint.nc")
        print("Saved to ERA5_monthly_means_gridpoint.nc")