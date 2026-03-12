import pandas as pd

print("Task 2 - Data Exploration Started")

data = {
    "temperature": [22, 24, 19, 23, None],
    "humidity": [55, 60, 58, None, 65],
    "wind_speed": [12, 14, 10, 11, 13]
}

df = pd.DataFrame(data)

print("\nFirst rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nStatistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())