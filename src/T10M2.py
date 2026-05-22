# ============================================
# Task 10: Cluster Execution -- Client Mode
# Author: Abdulatif Alabdulatif (ID:200291)
# ============================================

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SE446_M2_Task_10_Abdulatif") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("Master:", spark.sparkContext.master)

file_path = "hdfs:///data/chicago_crimes.csv"

df = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)

print("Real row count:", df.count())

df.printSchema()
df.show(5, truncate=False)