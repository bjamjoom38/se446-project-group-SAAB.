from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("CrimeTrends").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df = spark.read.csv("hdfs:///data/chicago_crimes.csv", header=True, inferSchema=True)
yearly = df.groupBy("Year").count().orderBy("Year")

print("=" * 35)
print("   Crime Count Per Year")
print("=" * 35)
yearly.show(50, truncate=False)

spark.stop()
