from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("ArrestRateAnalysis").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df = spark.read.csv("hdfs:///data/chicago_crimes.csv", header=True, inferSchema=True)

# Shows total arrest rate
total_crimes = df.count()
total_arrests = df.filter(F.col("Arrest") == True).count()
overall_rate = (total_arrests / total_crimes) * 100

print("=" * 40)
print("        Overall Arrest Rate")
print("=" * 40)
print(f"  Total Crimes  : {total_crimes:,}")
print(f"  Total Arrests : {total_arrests:,}")
print(f"  Arrest Rate   : {overall_rate:.2f}%")
print("=" * 40)

# Shows arrest rate per crime type
arrest_by_type = df.groupBy("Primary Type").agg(
    F.count("*").alias("Total Crimes"),
    F.sum(F.col("Arrest").cast("int")).alias("Total Arrests"),
).withColumn(
    "Arrest Rate (%)",
    F.round((F.col("Total Arrests") / F.col("Total Crimes")) * 100, 2),
).orderBy(F.col("Arrest Rate (%)").desc())

print("\n" + "=" * 60)
print("   Arrest Rate by Crime Type (Highest to Lowest)")
print("=" * 60)
arrest_by_type.show(10, truncate=False)

print("\n" + "=" * 60)
print("   Crime Types with Lowest Arrest Rates")
print("=" * 60)
arrest_by_type.orderBy(F.col("Arrest Rate (%)").asc()).show(10, truncate=False)

spark.stop()
