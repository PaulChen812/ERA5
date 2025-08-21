
import pandas as pd
import numpy as np
from nclim_climatology import CLIMATOLOGY_MEAN, MONTHLY_CLIMATOLOGY

def calculate_variance_std(csv_file: str):
    df = pd.read_csv(csv_file)

    results = {}


    overall_diff = df["Temperature"] - CLIMATOLOGY_MEAN
    results["overall"] = {
        "variance": float(np.var(overall_diff, ddof=0)),
        "std_dev": float(np.std(overall_diff, ddof=0)),
    }

    for month in range(1, 13):
        month_df = df[df["Month"] == month]
        if not month_df.empty:
            diff = month_df["Temperature"] - MONTHLY_CLIMATOLOGY[month]
            results[month] = {
                "variance": float(np.var(diff, ddof=0)),
                "std_dev": float(np.std(diff, ddof=0)),
            }

    return results


if __name__ == "__main__":
    stats = calculate_variance_std("nclim_averages.csv")


    print(f"Overall: variance={stats['overall']['variance']:.3f}, std={stats['overall']['std_dev']:.3f}")
    for month in range(1, 13):
        v = stats[month]["variance"]
        s = stats[month]["std_dev"]
        print(f"Month {month:02d}: variance={v:.3f}, std={s:.3f}")
