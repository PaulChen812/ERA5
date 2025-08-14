import os
import calendar
import xarray as xr
from dask.diagnostics import ProgressBar

class ClimatologyMeanCalculator:
    def __init__(self, root_dir, start_year=2010, end_year=2020):
        self.root_dir = root_dir
        self.start_year = start_year
        self.end_year = end_year
        self.file_list = self._gather_file_paths()

    def _gather_file_paths(self):
        # Efficiently generate only valid file paths
        return [
            os.path.join(self.root_dir, f"{year}", f"{month:02d}", f"t2m_{year}_{month:02d}_{day:02d}.nc")
            for year in range(self.start_year, self.end_year + 1)
            for month in range(1, 13)
            for day in range(1, calendar.monthrange(year, month)[1] + 1)
        ]

    def __call__(self, output_path="climatology_mean_2010_2020.nc"):
        print(f"Loading {len(self.file_list)} files from {self.start_year} to {self.end_year}...")

        # Open all NetCDF files lazily using Dask
        ds = xr.open_mfdataset(
            self.file_list,
            combine='by_coords',
            parallel=True,
            chunks={'time': 50}
        )

        # Compute the mean over time
        with ProgressBar():
            climatology_mean = ds['t2m'].mean(dim='time').compute()

        # Convert Kelvin to Celsius
        climatology_mean_C = climatology_mean - 273.15

        # Save as a new dataset
        output_ds = climatology_mean_C.to_dataset(name='t2m')
        output_ds.to_netcdf(output_path)
        print(f"Saved climatology mean to {output_path}")

        return output_ds

if __name__ == "__main__":
    calc = ClimatologyMeanCalculator("/global/cfs/cdirs/m3638/ERA5_Data/2m_temperature_daily")
    calc()
