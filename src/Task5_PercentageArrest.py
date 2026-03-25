import pandas as pd

# After filtering the data set I have downloaded it from "https://data.cityofchicago.org/Public-Safety/Chicago-CrimesSE446/wp32-6m43/revisions/0/soql"
df = pd.read_csv('Chicago_Crimes.csv')

# Count total crimes
total_crimes = len(df)

# Count crimes with arrest = True
arrests = df['Arrest'].sum()   # True = 1, False = 0

# Percentage
percentage = (arrests / total_crimes) * 100

# Print result
print(f"Total Crimes: {total_crimes}")
print(f"Crimes with Arrest: {arrests}")
print(f"Percentage of Crimes Resulting in Arrest: {percentage:.2f}%")