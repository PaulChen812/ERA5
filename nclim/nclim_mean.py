import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

class NClimGridTemperatureCNS:
    def __init__(self):
        self.base_url = "ftp://ftp.ncei.noaa.gov/pub/data/daily-grids/v1-0-0/averages/"
        self.years = range(1951, 2026)
        self.months = range(1, 13)

    def load_cns_file(self, year: int, month: int) -> float:
        yyyymm = f"{year}{month:02d}"
        file_name = f"tavg-{yyyymm}-cns-scaled.csv"
        url = f"{self.base_url}{year}/{file_name}"

        try:
            df = pd.read_csv(url, header=None)
        except Exception as e:
            print(f"Error loading {url}: {e}")
            return None

        numeric_columns = df.iloc[:, 6:]
        float_numbers = numeric_columns.values.flatten()
        float_numbers = float_numbers[float_numbers != -999.99]

        if len(float_numbers) == 0:
            return None
        return float_numbers.mean()

    def __call__(self, year: int, month: int):
        avg_temp = self.load_cns_file(year, month)
        if avg_temp is not None:
            x_val = year + (month - 1) / 12
            return (x_val, year, month, avg_temp)
        return None

    def load_all_batched(self, max_workers=4):
        all_results = []
        for year in self.years:
            print(f"Processing year {year}...")
            year_results = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self, year, m): m for m in self.months}
                for future in as_completed(futures):
                    res = future.result()
                    month = futures[future]
                    if res is not None:
                        year_results.append(res)
                        print(f"Finished {year}-{month:02d}")
            year_results.sort(key=lambda x: x[0])
            all_results.extend(year_results)

        all_results.sort(key=lambda x: x[0])
        return all_results

if __name__ == "__main__":
    nclim = NClimGridTemperatureCNS()
    results = nclim.load_all_batched(max_workers=4)

    df_monthly = pd.DataFrame(results, columns=["X", "Year", "Month", "Temperature"])
    df_monthly = df_monthly.drop(columns=["X"])
    df_monthly.to_csv("nclim_averages.csv", index=False)
    print("Saved monthly averages to nclim_averages.csv")

    df_yearly = df_monthly.groupby("Year", as_index=False)["Temperature"].mean()
    df_yearly.to_csv("nclim_yearly_mean.csv", index=False)
    print("Saved yearly means to nclim_yearly_mean.csv")
