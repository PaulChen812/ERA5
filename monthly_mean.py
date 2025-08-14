import os
from pathlib import Path
import xarray as xr
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

MT = 273.15
EARTH_RADIUS = 6371000

class MonthlyMeanCalculator:
    def __init__(self, directory, variable, latitude_range=None):
        self.directory = Path(directory)
        self.variable = variable
        self.latitude_range = latitude_range
        self.area_da = None
        self.initialize_area()

    def __call__(self, year: int, month: int) -> float:

        monthly_mean_da = self.monthly_mean(year, month)
        return self.area_weighted_mean(monthly_mean_da)

    def _lat_slice(self, lat_values: np.ndarray) -> slice:

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

        if var_sel.sizes.get("latitude", 0) < 2 or var_sel.sizes.get("longitude", 0) < 2:
            ds.close()
            raise RuntimeError(
                f"After latitude selection, grid is too small: "
                f"lat={var_sel.sizes.get('latitude')}, lon={var_sel.sizes.get('longitude')}. "
                f"Check latitude_range={self.latitude_range} vs dataset ordering."
            )

        if self.area_da is None:
            self.area_da = self.calculate_area(
                var_sel.latitude.values, var_sel.longitude.values
            )

        daily_mean = var_sel.mean(dim="time")
        ds.close()
        return daily_mean

    def initialize_area(self):
        sample_file = sorted(Path(self.directory).rglob("*.nc"))[0]
        ds = xr.open_dataset(sample_file)
        self.area_da = self.calculate_area(ds)
        ds.close()

    def calculate_area(self, ds):
        R = 6371.0
        lat_rad = np.deg2rad(ds.latitude)
        lon_rad = np.deg2rad(ds.longitude)
        dlat = np.abs(lat_rad[1] - lat_rad[0])
        dlon = np.abs(lon_rad[1] - lon_rad[0])
        area = (R**2) * dlat * dlon * np.cos(lat_rad)
        area_da = xr.DataArray(
            np.repeat(area.values[:, np.newaxis], len(lon_rad), axis=1),
            coords=[ds.latitude, ds.longitude],
            dims=["latitude", "longitude"]
        )
        return area_da

    def area_weighted_mean(self, temp_da: xr.DataArray) -> float:
        if self.area_da is None:
            raise RuntimeError("Run preprocess() at least once so area grid is initialized.")
        total_area = self.area_da.sum(dim=["latitude", "longitude"])
        temp_weighted_sum = (temp_da * self.area_da).sum(dim=["latitude", "longitude"])
        return (temp_weighted_sum / total_area).item()

    def monthly_mean(self, year: int, month: int) -> xr.DataArray:
        month_dir = self.directory / f"{year:04d}" / f"{month:02d}"
        files = sorted(month_dir.glob("*.nc"))
        if not files:
            raise FileNotFoundError(f"No .nc files found in {month_dir}")

        total = None
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.preprocess, f): f for f in files}
            for future in as_completed(futures):
                dm = future.result()
                total = dm if total is None else total + dm

        monthly_mean_da = total / len(files)
        return monthly_mean_da


if __name__ == "__main__":
    directory = "/global/cfs/cdirs/m3638/ERA5_Data/2m_temperature_daily/"
    calc = MonthlyMeanCalculator(
        directory=directory, variable="t2m", latitude_range=(25, 50)
    )

    years = range(1941, 2026)
    months = range(1, 13)
    year_month_pairs = list(itertools.product(years, months))

    results = {}

    max_workers = 8
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(calc, y, m): (y, m) for y, m in year_month_pairs}
        for future in as_completed(futures):
            y, m = futures[future]
            try:
                mean_temp = future.result()
                results[(y, m)] = mean_temp
                print(f"Finished calculating {y}-{m:02d}: {mean_temp:.2f} °C")
            except FileNotFoundError:
                print(f"No data for {y}-{m:02d}")
            except Exception as e:
                print(f"Error for {y}-{m:02d}: {e}")


    times = [np.datetime64(f"{y}-{m:02d}-15") for y, m in sorted(results.keys())]
    data = [results[k] for k in sorted(results.keys())]
    da = xr.DataArray(
        data, dims=["time"], coords={"time": times}, name="ERA5_monthly_mean"
    )
    da.to_netcdf("ERA_Monthly_mean.nc")
    print("Saved ERA_Monthly_mean.nc")
