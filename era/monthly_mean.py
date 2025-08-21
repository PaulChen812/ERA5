import os
from pathlib import Path
import xarray as xr
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import regionmask
import cartopy.io.shapereader as shpreader

MT = 273.15
EARTH_RADIUS = 6371000.0  

def _wrap_to_180(lon_array):
    """Wrap longitudes to [-180, 180)."""
    return ((np.asarray(lon_array) + 180.0) % 360.0) - 180.0

def create_us_mask(lat, lon):
    """
    Creates a mask for the contiguous United States.

    Fix: ERA5 longitudes are often 0..360. We wrap them to -180..180 for
    geometric tests against Natural Earth polygons and for the CONUS lon box.
    """

    lon_for_test = _wrap_to_180(lon)
    lat_for_test = np.asarray(lat)


    shpfilename = shpreader.natural_earth(
        resolution="50m",
        category="cultural",
        name="admin_0_countries",
    )
    reader = shpreader.Reader(shpfilename)
    countries = list(reader.records())

    us_poly = [c.geometry for c in countries if c.attributes["ADMIN"] == "United States of America"]
    if not us_poly:
        raise RuntimeError("United States polygon not found in Natural Earth file.")


    mask = regionmask.Regions([us_poly[0]])
    mask_2d = mask.mask(lon_for_test, lat_for_test) 


    lat2d, lon2d = np.meshgrid(lat_for_test, lon_for_test, indexing="ij")


    conus_box = (
        (lat2d >= 24.5) & (lat2d <= 49.5) &  
        (lon2d >= -125.0) & (lon2d <= -66.5)  
    )


    contiguous_mask = (mask_2d == 0) & conus_box


    return contiguous_mask


def compute_gridcell_area(lat_vals, lon_vals, R=EARTH_RADIUS):
    """
    Compute the area of spherical rectangles for most grid cells, but handle polar cells as spherical triangles.
    """
    lat_rad = np.deg2rad(lat_vals)
    lon_rad = np.deg2rad(lon_vals)

    dlat = np.abs(lat_rad[1] - lat_rad[0])
    dlon = np.abs(lon_rad[1] - lon_rad[0])


    area = np.zeros((len(lat_rad), len(lon_rad)))

    for i, phi in enumerate(lat_rad):

        phi1 = phi - dlat / 2
        phi2 = phi + dlat / 2

        if phi2 > np.pi / 2:  

            cap_area = 2 * np.pi * R**2 * (1 - np.sin(phi - dlat / 2))
            area[i, :] = cap_area / len(lon_rad)  
        elif phi1 < -np.pi / 2:  
            cap_area = 2 * np.pi * R**2 * (1 + np.sin(phi + dlat / 2))
            area[i, :] = cap_area / len(lon_rad)
        else:
            
            area[i, :] = (R**2) * dlon * (np.sin(phi2) - np.sin(phi1))

    return area


class MonthlyUSMeanCalculator:
    def __init__(self, directory, variable, latitude_range=None):
        self.directory = Path(directory)
        self.variable = variable
        self.latitude_range = latitude_range
        self.area_da = None
        self.us_mask = None
        self.initialize_area_and_mask()

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

        daily_mean = var_sel.mean(dim="time")

        daily_mean = daily_mean.where(self.us_mask)

        ds.close()
        return daily_mean

    def initialize_area_and_mask(self):
        sample_file = sorted(Path(self.directory).rglob("*.nc"))[0]
        ds = xr.open_dataset(sample_file)
        lat_vals = ds.latitude.values
        lon_vals = ds.longitude.values

        area = compute_gridcell_area(lat_vals, lon_vals, R=EARTH_RADIUS)
        self.area_da = xr.DataArray(
            area,
            coords=[lat_vals, lon_vals],
            dims=["latitude", "longitude"],
        )

        self.us_mask = create_us_mask(lat_vals, lon_vals)
        self.us_mask = xr.DataArray(
            self.us_mask, coords=[lat_vals, lon_vals], dims=["latitude", "longitude"]
        )

        ds.close()

    def area_weighted_mean(self, temp_da: xr.DataArray) -> float:
        area_da_masked = self.area_da.where(self.us_mask)
        temp_da_masked = temp_da.where(self.us_mask)

        total_area = area_da_masked.sum(dim=["latitude", "longitude"])
        temp_weighted_sum = (temp_da_masked * area_da_masked).sum(dim=["latitude", "longitude"])

        return (temp_weighted_sum / total_area).item()

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
        return monthly_mean_da


if __name__ == "__main__":
    directory = "/global/cfs/cdirs/m3638/ERA5_Data/2m_temperature_daily/"
    calc = MonthlyUSMeanCalculator(
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
                print(f"Finished calculating US mean {y}-{m:02d}: {mean_temp:.2f} °C")
            except FileNotFoundError:
                print(f"No data for {y}-{m:02d}")
            except Exception as e:
                print(f"Error for {y}-{m:02d}: {e}")


    years_out = []
    months_out = []
    temps_out = []
    for (y, m) in sorted(results.keys()):
        years_out.append(y)
        months_out.append(m)
        temps_out.append(results[(y, m)])


    ds = xr.Dataset(
        {
            "Temperature": (("time",), temps_out),
            "Year": (("time",), years_out),
            "Month": (("time",), months_out),
        },
        coords={"time": np.arange(len(temps_out))},
    )

    ds.to_netcdf("ERA_US_Monthly_mean.nc")
    print("Saved ERA_US_Monthly_mean.nc")
