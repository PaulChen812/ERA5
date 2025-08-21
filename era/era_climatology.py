import xarray as xr
import numpy as np

class ERAClimatology:
    def __init__(self, nc_file: str, var_name: str = "Temperature"):
        self.ds = xr.open_dataset(nc_file)
        self.var_name = var_name
        self.data = self.ds[self.var_name]


        if "Year" in self.ds and "Month" in self.ds:
            self.years = self.ds["Year"]
            self.months = self.ds["Month"]
        else:
            raise ValueError("NetCDF file must contain Year and Month variables")

        self.CLIMATOLOGY_MEAN = None
        self.MONTHLY_CLIMATOLOGY = {}

        self._compute_climatology()

    def _compute_climatology(self):

        self.CLIMATOLOGY_MEAN = float(self.data.mean().values)


        for m in range(1, 13):
            monthly_data = self.data.where(self.months == m, drop=True)
            if monthly_data.size > 0:
                self.MONTHLY_CLIMATOLOGY[m] = float(monthly_data.mean().values)
            else:
                self.MONTHLY_CLIMATOLOGY[m] = np.nan



_clim = ERAClimatology("ERA_US_Monthly_mean.nc")
CLIMATOLOGY_MEAN = _clim.CLIMATOLOGY_MEAN
MONTHLY_CLIMATOLOGY = _clim.MONTHLY_CLIMATOLOGY

if __name__ == "__main__":
    print("Overall Climatology Mean:", CLIMATOLOGY_MEAN)
    print("Monthly Climatology Means:", MONTHLY_CLIMATOLOGY)
