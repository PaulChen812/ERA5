import os
from pathlib import Path
import xarray as xr
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

MT = 273.15 

class MonthlyGridMeanCalculator:
    def __init__(self, directory, variable, latitude_range=None):
        self.directory = Path(directory)
        self.variable = variable
        self.latitude_range = latitude_range

    def __call__(self, year: int, month: int) -> xr.DataArray:
        return self.monthly_mean(year, month)

    def _lat_slice(self, lat_values: np.ndarray) -> slice:
        if self.latitude_range is None:
            return slice(None)
        lo, hi = self.latitude_range
        lo, hi = float(lo), float(hi)
        first, last = float(lat_values[0]), float(lat_values[-1])
        if first > last:
            start, stop = max(lo, hi), min(lo, hi)
        else:
            start, stop = min(lo, hi), max(lo, hi)
        return slice(start, stop)

    def preprocess(self, filename: Path) -> xr.DataArray:
        ds = xr.open_dataset(filename)
        if self.variable not in ds:
            ds.close()
            raise RuntimeError(f"Variable '{self.variable}' not found in dataset.")
        lat_vals = ds["latitude"].values
        lat_slice = self._lat_slice(lat_vals)
        var_sel = ds[self.variable].sel(latitude=lat_slice) - MT
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
        directory=directory, variable="t2m", latitude_range=(25, 50)
    )

    years = range(1950, 2025)
    months = range(1, 13)
    year_month_pairs = list(itertools.product(years, months))

    results = []

    max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    print(f"Using max_workers = {max_workers}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(calc, y, m): (y, m) for y, m in year_month_pairs}
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
        print("Saved ERA5_monthly_means_gridpoint.nc (sorted)")
