import pandas as pd
import matplotlib.pyplot as plt


df_yearly = pd.read_csv("nclim_yearly_mean.csv")

plt.figure(figsize=(16,6))
plt.plot(df_yearly['Year'], df_yearly['Temperature'], marker='o', linestyle='-', markersize=4, label='Yearly Avg')
plt.ylim(10,14)
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.title("NClimGrid Daily - Yearly Mean Temperature (CNS - Contiguous U.S.)")
plt.grid(True)
plt.xlim(1950, 2030)
plt.xticks(range(1950, 2031, 10))
plt.legend()
plt.tight_layout()
plt.savefig("nclim_yearly_means.png", dpi=300)
plt.show()
