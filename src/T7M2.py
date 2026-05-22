# ============================================
# Task 7: Feature Importances & Interpretation
# Author: Abdullah Almutabagani (ID: 220534)
# ============================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.ml.classification import RandomForestClassifier

from m2_spark_ml import (
    feature_pipeline,
    load_data,
    prepare_data,
    build_spark,
    print_feature_importances,
)


def main():
    spark = build_spark()
    try:
        df = prepare_data(load_data(spark)).cache()
        print(f"Prepared ML rows: {df.count():,}")

        train_df, _ = df.randomSplit([0.8, 0.2], seed=42)
        train_df = train_df.cache()
        print(f"Training rows: {train_df.count():,}")

        rf = RandomForestClassifier(
            labelCol="label",
            featuresCol="features",
            numTrees=100,
            maxDepth=5,
            seed=42,
        )
        model = feature_pipeline(rf).fit(train_df)
        print_feature_importances(model)

        train_df.unpersist()
        df.unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
