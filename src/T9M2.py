# ============================================
# Task 9: Local Execution Setup
# Author: Abdulatif Alabdulatif (ID: 200291)
# ============================================

from pyspark.sql import SparkSession, Row
import random

spark = SparkSession.builder \
    .appName("Chicago Crimes M2 Local") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("Master:", spark.sparkContext.master)

random.seed(42)

crime_types = [
    "THEFT", "BATTERY", "CRIMINAL DAMAGE", "NARCOTICS",
    "ASSAULT", "BURGLARY", "ROBBERY", "MOTOR VEHICLE THEFT",
    "WEAPONS VIOLATION", "PROSTITUTION"
]

locations = [
    "STREET", "RESIDENCE", "APARTMENT", "SIDEWALK",
    "PARKING LOT/GARAGE", "SCHOOL", "RESTAURANT", "STORE"
]

rows = []

for i in range(10000):
    year = random.randint(2015, 2024)
    hour = random.randint(0, 23)
    crime_type = random.choice(crime_types)
    location = random.choice(locations)
    domestic = random.choice([True, False])
    district = random.randint(1, 25)

    # Simple realistic arrest pattern
    if crime_type in ["NARCOTICS", "WEAPONS VIOLATION", "PROSTITUTION"]:
        arrest = random.random() < 0.70
    else:
        arrest = random.random() < 0.20

    rows.append(Row(
        ID=i + 1,
        Date=f"01/01/{year} {hour:02d}:00:00",
        **{
            "Primary Type": crime_type,
            "Location Description": location,
            "Arrest": arrest,
            "Domestic": domestic,
            "District": district,
            "Year": year,
            "Hour": hour
        }
    ))

df = spark.createDataFrame(rows)

print("Generated row count:", df.count())
df.show(5, truncate=False)