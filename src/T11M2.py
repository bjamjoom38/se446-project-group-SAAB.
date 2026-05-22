from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .appName("SE446_M2_Task_11_Abdulatif") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df = spark.read.csv("hdfs:///data/chicago_crimes.csv", header=True, inferSchema=True)
df = df.withColumn("Arrest", F.col("Arrest").cast("int"))
df = df.dropna(subset=["Primary Type", "Location Description", "Arrest", "Year"])

pipeline = Pipeline(stages=[
    StringIndexer(inputCol="Primary Type", outputCol="PrimaryTypeIndex", handleInvalid="keep"),
    StringIndexer(inputCol="Location Description", outputCol="LocationIndex", handleInvalid="keep"),
    VectorAssembler(inputCols=["PrimaryTypeIndex", "LocationIndex", "Year"], outputCol="features"),
    RandomForestClassifier(labelCol="Arrest", featuresCol="features", numTrees=20, maxDepth=5, seed=42),
])

train, test = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train)
predictions = model.transform(test)

auc = BinaryClassificationEvaluator(
    labelCol="Arrest",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC",
).evaluate(predictions)

print("=" * 40)
print(f"  AUC-ROC Score: {auc:.4f}")
print("=" * 40)
predictions.groupBy("Arrest", "prediction").count().orderBy("Arrest", "prediction").show()

spark.stop()
