import pandas as pd
import matplotlib.pyplot as plt

# After filtering the data set I have downloaded it from "https://data.cityofchicago.org/Public-Safety/Chicago-CrimesSE446/wp32-6m43/revisions/0/soql"
df = pd.read_csv('Chicago_Crimes.csv')

# Parse dates (FAST version)
print("Parsing dates...")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

# Extract year
print("Extracting year...")
df['Year'] = df['Date'].dt.year

# Count crimes per year
print("Counting crimes...")
crime_per_year = df['Year'].value_counts().sort_index()

# Ensure full range (2001–2026)
all_years = range(2001, 2027)
crime_per_year = crime_per_year.reindex(all_years, fill_value=0)

crime_per_year.index = crime_per_year.index.astype(int)

# Plot
print("Plotting...")
plt.figure(figsize=(12, 6))
crime_per_year.plot(kind='line', marker='o')

plt.xticks(all_years, rotation=45)
plt.title('Total Crimes Per Year (2001–2026)')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.grid()

plt.tight_layout()
plt.show()

print("DONE")