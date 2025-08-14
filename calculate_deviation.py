import os
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar

class MonthlyDeviationCalculator:
    def __init__(self, month_dir, climatology_file="climatology_mean_2010_2020.nc"):
        self.month_dir = month_dir
        self.climatology_file = climatology_file
        self.file_list = self._gather_file_paths()

    def _gather_file_paths(self):
        return [
            os.path.join(self.month_dir, fname)
            for fname in sorted(os.listdir(self.month_dir))
            if fname.endswith(".nc") and fname.startswith("t2m")
        ]
        
    def __call__(self, output_path="monthly_deviation.nc"):
        print(f"Loading {len(self.file_list)} files from {self.month_dir}...")

        ds_month = xr.open_mfdataset(
            self.file_list,
            combine='by_coords',
            parallel=True,
            chunks={'time': 10}
        )

        ds_clim = xr.open_dataset(self.climatology_file)
        clim_mean = ds_clim['t2m'] 

        t2m_month = ds_month['t2m'] - 273.15

        diff = t2m_month - clim_mean

        with ProgressBar():
            variance = (diff ** 2).mean(dim='time').compute()
            std_dev = np.sqrt(variance)

        result_ds = xr.Dataset({
            'variance': variance,
            'std_dev': std_dev,
        })
        
        result_ds.to_netcdf(output_path)
        return result_ds

if __name__ == "__main__":
    month_dir = "/global/cfs/cdirs/m3638/ERA5_Data/2m_temperature_daily/2018/07"
    calc = MonthlyDeviationCalculator(month_dir)
    calc()