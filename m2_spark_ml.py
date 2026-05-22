# ============================================
# SE446 - Milestone 2: Spark ML Pipeline
# Group SAAB
#
# Tasks 5-7: Spark ML feature engineering, model evaluation,
#            and Random Forest interpretation
# ============================================

import os
import random
import time
from datetime import datetime, timedelta

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


FEATURE_NAMES = ["District", "crime_index", "Hour", "domestic_index"]


def build_spark():
    master = os.environ.get("SPARK_MASTER")
    builder = SparkSession.builder.appName("SE446-M2-GroupSAAB-ML")
    if master:
        builder = builder.master(master)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark {spark.version} | Master: {spark.sparkContext.master}")
    return spark


def generate_local_data(spark, rows_count=10000):
    random.seed(42)

    schema = StructType([
        StructField("ID", IntegerType(), False),
        StructField("Date", StringType(), False),
        StructField("Primary Type", StringType(), False),
        StructField("Location Description", StringType(), False),
        StructField("Arrest", BooleanType(), False),
        StructField("Domestic", BooleanType(), False),
        StructField("District", IntegerType(), False),
        StructField("Year", IntegerType(), False),
        StructField("Hour", IntegerType(), False),
    ])

    crime_types = [
        "THEFT", "BATTERY", "CRIMINAL DAMAGE", "NARCOTICS", "ASSAULT",
        "OTHER OFFENSE", "BURGLARY", "MOTOR VEHICLE THEFT", "ROBBERY",
        "DECEPTIVE PRACTICE", "CRIMINAL TRESPASS", "WEAPONS VIOLATION",
        "PROSTITUTION", "HOMICIDE", "ARSON",
    ]
    crime_weights = [
        0.20, 0.14, 0.10, 0.08, 0.07, 0.07, 0.06, 0.05, 0.04,
        0.04, 0.04, 0.03, 0.02, 0.02, 0.04,
    ]
    locations = [
        "STREET", "RESIDENCE", "APARTMENT", "SIDEWALK", "OTHER",
        "PARKING LOT/GARAGE(NON.RESID.)", "ALLEY",
        "SCHOOL, PUBLIC, BUILDING", "RESTAURANT", "GAS STATION",
    ]

    rows = []
    for i in range(rows_count):
        year = random.randint(2015, 2024)
        hour = random.randint(0, 23)
        date = datetime(year, 1, 1) + timedelta(
            days=random.randint(0, 364),
            hours=hour,
            minutes=random.randint(0, 59),
        )
        crime_type = random.choices(crime_types, weights=crime_weights, k=1)[0]
        domestic = random.choice([True, False])
        district = random.randint(1, 25)

        base_rate = 0.20
        if crime_type in {"NARCOTICS", "WEAPONS VIOLATION", "PROSTITUTION"}:
            base_rate = 0.70
        elif crime_type in {"HOMICIDE", "ROBBERY"}:
            base_rate = 0.35
        if domestic:
            base_rate += 0.05

        rows.append((
            i + 1,
            date.strftime("%m/%d/%Y %I:%M:%S %p"),
            crime_type,
            random.choice(locations),
            random.random() < min(base_rate, 0.95),
            domestic,
            district,
            year,
            hour,
        ))

    df = spark.createDataFrame(rows, schema)
    print(f"Generated local dataset: {df.count():,} rows")
    return df


def load_data(spark):
    data_path = os.environ.get("M2_DATA_PATH")

    if data_path:
        df = spark.read.csv(data_path, header=True, inferSchema=True)
        print(f"Loaded dataset from {data_path}: {df.count():,} rows")
        return df

    on_cluster = (
        os.environ.get("HADOOP_HOME") is not None
        or os.environ.get("YARN_CONF_DIR") is not None
        or spark.sparkContext.master.startswith("yarn")
    )
    if on_cluster:
        data_path = "hdfs:///data/chicago_crimes.csv"
        df = spark.read.csv(data_path, header=True, inferSchema=True)
        print(f"Loaded HDFS dataset from {data_path}: {df.count():,} rows")
        return df

    return generate_local_data(spark)


def prepare_data(df):
    if "Hour" not in df.columns:
        parsed_date = F.coalesce(
            F.to_timestamp(F.col("Date"), "MM/dd/yyyy hh:mm:ss a"),
            F.to_timestamp(F.col("Date"), "MM/dd/yyyy HH:mm:ss"),
        )
        df = df.withColumn("Hour", F.hour(parsed_date))

    label_expr = (
        F.when(F.lower(F.col("Arrest").cast("string")) == "true", F.lit(1))
         .when(F.lower(F.col("Arrest").cast("string")) == "false", F.lit(0))
         .otherwise(F.col("Arrest").cast("int"))
    )

    return (
        df.withColumn("label", label_expr)
          .withColumn("DomesticStr", F.col("Domestic").cast("string"))
          .withColumn("District", F.col("District").cast("double"))
          .withColumn("Hour", F.col("Hour").cast("double"))
          .dropna(subset=["Primary Type", "DomesticStr", "District", "Hour", "label"])
    )


def feature_pipeline(classifier):
    crime_indexer = StringIndexer(
        inputCol="Primary Type",
        outputCol="crime_index",
        handleInvalid="skip",
    )
    domestic_indexer = StringIndexer(
        inputCol="DomesticStr",
        outputCol="domestic_index",
        handleInvalid="skip",
    )
    assembler = VectorAssembler(
        inputCols=FEATURE_NAMES,
        outputCol="features",
        handleInvalid="skip",
    )
    return Pipeline(stages=[crime_indexer, domestic_indexer, assembler, classifier])


def confusion_counts(predictions):
    rows = (
        predictions.groupBy("label", "prediction")
        .count()
        .collect()
    )
    counts = {(int(row["label"]), int(row["prediction"])): row["count"] for row in rows}
    return {
        "TN": counts.get((0, 0), 0),
        "FP": counts.get((0, 1), 0),
        "FN": counts.get((1, 0), 0),
        "TP": counts.get((1, 1), 0),
    }


def evaluate_predictions(name, predictions, training_time):
    auc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    ).evaluate(predictions)
    accuracy = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy",
    ).evaluate(predictions)
    f1 = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1",
    ).evaluate(predictions)
    precision = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision",
    ).evaluate(predictions)
    recall = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall",
    ).evaluate(predictions)
    confusion = confusion_counts(predictions)

    return {
        "Model": name,
        "AUC": auc,
        "Accuracy": accuracy,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Training Seconds": training_time,
        **confusion,
    }


def train_and_evaluate(train_df, test_df):
    models = [
        (
            "Logistic Regression",
            LogisticRegression(
                labelCol="label",
                featuresCol="features",
                maxIter=100,
                regParam=0.01,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                labelCol="label",
                featuresCol="features",
                numTrees=100,
                maxDepth=5,
                maxBins=64,
                seed=42,
            ),
        ),
        (
            "GBT",
            GBTClassifier(
                labelCol="label",
                featuresCol="features",
                maxIter=50,
                maxDepth=5,
                maxBins=64,
                seed=42,
            ),
        ),
    ]

    results = []
    trained = {}
    for name, classifier in models:
        print("\n" + "=" * 70)
        print(f"Training {name}")
        print("=" * 70)
        pipeline = feature_pipeline(classifier)
        start = time.time()
        model = pipeline.fit(train_df)
        training_time = time.time() - start
        predictions = model.transform(test_df).cache()
        predictions.count()
        result = evaluate_predictions(name, predictions, training_time)
        results.append(result)
        trained[name] = model

        print(
            f"AUC={result['AUC']:.4f} "
            f"Accuracy={result['Accuracy']:.4f} "
            f"F1={result['F1']:.4f} "
            f"Precision={result['Precision']:.4f} "
            f"Recall={result['Recall']:.4f} "
            f"Time={result['Training Seconds']:.1f}s"
        )
        print(
            f"Confusion Matrix: TN={result['TN']} FP={result['FP']} "
            f"FN={result['FN']} TP={result['TP']}"
        )
        predictions.unpersist()

    return results, trained


def print_results_table(results):
    print("\n" + "=" * 118)
    print("Task 6: Model Comparison")
    print("=" * 118)
    header = (
        f"{'Model':<22} {'AUC':>7} {'Acc':>7} {'F1':>7} {'Prec':>7} "
        f"{'Recall':>7} {'Seconds':>9} {'TN':>9} {'FP':>9} {'FN':>9} {'TP':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['Model']:<22} {row['AUC']:>7.4f} {row['Accuracy']:>7.4f} "
            f"{row['F1']:>7.4f} {row['Precision']:>7.4f} {row['Recall']:>7.4f} "
            f"{row['Training Seconds']:>9.1f} {row['TN']:>9} {row['FP']:>9} "
            f"{row['FN']:>9} {row['TP']:>9}"
        )


def print_feature_importances(model):
    rf_model = model.stages[-1]
    importances = list(zip(FEATURE_NAMES, rf_model.featureImportances.toArray()))
    importances.sort(key=lambda item: item[1], reverse=True)

    print("\n" + "=" * 70)
    print("Task 7: Random Forest Feature Importances")
    print("=" * 70)
    for feature, importance in importances:
        bar = "#" * int(round(importance * 50))
        print(f"{feature:<16} {importance:>8.4f} {bar}")

    print("\nInterpretation:")
    print(
        "Higher importance means the Random Forest used that feature more often "
        "to split arrest and non-arrest cases. Crime type usually dominates "
        "because arrest patterns differ strongly across categories such as "
        "narcotics, weapons violations, theft, and burglary."
    )


def main():
    spark = build_spark()
    try:
        df = prepare_data(load_data(spark)).cache()
        usable_rows = df.count()
        print(f"Prepared ML rows: {usable_rows:,}")

        print("\n" + "=" * 70)
        print("Task 5: Feature Engineering Pipeline")
        print("=" * 70)
        preview_model = feature_pipeline(
            RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=5, seed=42)
        ).fit(df.limit(1000))
        preview_model.transform(df).select(
            "District", "Primary Type", "Hour", "Domestic", "label", "features"
        ).show(5, truncate=False)
        print("Feature vector positions: [District, crime_index, Hour, domestic_index]")

        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        train_df = train_df.cache()
        test_df = test_df.cache()
        print(f"Train rows: {train_df.count():,} | Test rows: {test_df.count():,}")

        results, trained = train_and_evaluate(train_df, test_df)
        print_results_table(results)
        print_feature_importances(trained["Random Forest"])

        train_df.unpersist()
        test_df.unpersist()
        df.unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
