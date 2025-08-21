import pandas as pd
from pathlib import Path

file = Path("nclim_averages.csv")

def calculate_climatology(file_path: Path = file) -> float:
    df = pd.read_csv(file_path)
    climatology_mean = df['Temperature'].mean()

    monthly_means = df.groupby('Month')['Temperature'].mean()
    return climatology_mean, monthly_means


CLIMATOLOGY_MEAN, MONTHLY_CLIMATOLOGY = calculate_climatology()

if __name__ == "__main__":
    print(f"Overall Climatology Mean (1951–2025): {CLIMATOLOGY_MEAN:.3f} °C\n")
    print("Monthly Climatology Means (1951–2025):")
    for month, value in MONTHLY_CLIMATOLOGY.items():
        print(f"  Month {month:2d}: {value:.3f} °C")