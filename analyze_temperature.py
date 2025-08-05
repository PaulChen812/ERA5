import os
import numpy as np
from netCDF4 import Dataset
import sys
import multiprocessing as mp
from scipy import stats

def process_file(filepath):
    try:
        with Dataset(filepath, 'r') as nc:
            t2m = nc.variables['t2m']
            scale_factor = t2m.scale_factor
            add_offset = t2m.add_offset
            fill_value = t2m._FillValue

            count = 0
            mean = 0.0
            M2 = 0.0
            global_min = float('inf')
            global_max = float('-inf')
            values = []

            for i in range(t2m.shape[0]):  # over time
                data = t2m[i, :, :].astype(np.float64)
                data[data == fill_value] = np.nan
                data = data * scale_factor + add_offset
                data = data - 273.15  # to Celsius
                data = data[~np.isnan(data)]

                if data.size == 0:
                    continue

                # For mean and std (Welford’s algorithm)
                for x in data:
                    count += 1
                    delta = x - mean
                    mean += delta / count
                    delta2 = x - mean
                    M2 += delta * delta2

                # For min/max/mode
                global_min = min(global_min, np.min(data))
                global_max = max(global_max, np.max(data))
                values.extend(data)

            std = np.sqrt(M2 / count) if count > 1 else np.nan
            median = np.median(values)
            mode = stats.mode(values, nan_policy='omit').mode.item()

            return mean, global_max, global_min, median, mode, std, count

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def combine_stats(results):
    """Combine running stats from each file"""
    total_count = 0
    total_mean = 0.0
    M2 = 0.0
    mins, maxs, medians, modes = [], [], [], []

    for res in results:
        if res is None:
            continue
        mean, max_val, min_val, median, mode, std, count = res
        if count == 0:
            continue

        # Welford's combination
        delta = mean - total_mean
        total = total_count + count
        total_mean += delta * count / total
        M2 += std**2 * (count - 1) + delta**2 * total_count * count / total

        total_count = total
        mins.append(min_val)
        maxs.append(max_val)
        medians.append(median)
        modes.append(mode)

    total_std = np.sqrt(M2 / total_count) if total_count > 1 else np.nan

    return {
        "Mean": total_mean,
        "Max": max(maxs),
        "Min": min(mins),
        "Median": np.median(medians),
        "Mode": stats.mode(modes, nan_policy='omit').mode.item(),
        "Std": total_std,
        "Count": total_count
    }

def main(directory):
    nc_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.nc')])
    if not nc_files:
        print("No .nc files found.")
        return

    print(f"Processing {len(nc_files)} files with {mp.cpu_count()} processes...")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_file, nc_files)

    stats_result = combine_stats(results)

    print("\n=== Monthly Stats (2m Temp, °C) ===")
    for key in ["Mean", "Max", "Min", "Median", "Mode", "Std"]:
        print(f"{key}: {stats_result[key]:.2f} °C")
    print(f"Total points processed: {stats_result['Count']:,}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_month.py <directory_path>")
    else:
        main(sys.argv[1])

