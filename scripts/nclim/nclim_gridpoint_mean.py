import os
from pathlib import Path
import xarray as xr
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

class NClimGridMonthlyAverager:
    def __init__(self, base_dir=None, years=None, months=None, output_file=None):
        self.base_dir = Path(base_dir or os.path.join(os.environ.get("SCRATCH", "/tmp"), "nclimgrid"))
        self.years = years or range(1951, 2025)
        self.months = months or range(1, 13)
        self.output_file = output_file or "nclim_gridpoint_mean.nc"

    def calc(self, year: int, month: int):
        """Compute monthly mean for a single month of a given year."""
        file_name = f"ncdd-{year}{month:02d}-grd-scaled.nc"
        file_path = self.base_dir / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            ds = xr.open_dataset(file_path)
        except Exception as e:
            raise RuntimeError(f"Error opening {file_path}: {e}")

        if "tavg" not in ds:
            raise RuntimeError(f"No 'tavg' variable in {file_path}")

        monthly_mean = ds["tavg"].mean(dim="time", skipna=True)
        arr = monthly_mean.values.astype(np.float32)
        lat = ds["lat"].values
        lon = ds["lon"].values
        ds.close()

        return year, month, arr, lat, lon

    def run(self):

        year_month_pairs = [(y, m) for y in self.years for m in self.months]
        results = []

        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
        print(f"Using max_workers = {max_workers}")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.calc, y, m): (y, m) for (y, m) in year_month_pairs}

            for future in as_completed(futures):
                y, m = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Finished {y}-{m:02d}")
                except FileNotFoundError:
                    print(f"Missing file for {y}-{m:02d}")
                except Exception as e:
                    print(f"Error for {y}-{m:02d}: {e}")

        if not results:
            print("No results found")
            return


        results.sort(key=lambda x: (x[0], x[1]))
        years = [r[0] for r in results]
        months = [r[1] for r in results]
        data = np.stack([r[2] for r in results])
        lat = results[0][3]
        lon = results[0][4]

        ds_out = xr.Dataset(
            {
                "tavg_mean": (("time", "lat", "lon"), data, {"units": "degree_Celsius"})
            },
            coords={
                "year": ("time", years),
                "month": ("time", months),
                "lat": lat,
                "lon": lon,
            },
        )

        ds_out.to_netcdf(self.output_file)
        print(f"🎉 Saved monthly mean fields to {self.output_file}")


if __name__ == "__main__":
    averager = NClimGridMonthlyAverager()
    averager.run()
