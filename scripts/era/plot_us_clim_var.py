import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

infile = "ERA5_US_monthly_clim_var.nc"

ds = xr.open_dataset(infile)


months = np.arange(1, 13)
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


clim_values = [ds[f"US_mean_{m.lower()}"].item() for m in month_names]
var_values = [ds[f"US_var_{m.lower()}"].item() for m in month_names]


plt.figure(figsize=(10,5))
plt.plot(months, clim_values, marker='o', label="Climatology Mean (°C)")
plt.plot(months, var_values, marker='s', label="Variance (°C²)")

plt.xticks(months, month_names)
plt.xlabel("Month")
plt.ylabel("Temperature / Variance")
plt.title("U.S. Area-Weighted Monthly Climatology and Variance (1950–2024)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("US_monthly_clim_var.png", dpi=300)
plt.show()
