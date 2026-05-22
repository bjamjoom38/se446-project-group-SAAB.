# ============================================
# Task 6: Train and Evaluate Three Models
# Author: Abdullah Almutabagani (ID: 220534)
# ============================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from m2_spark_ml import (
    load_data,
    prepare_data,
    build_spark,
    print_results_table,
    train_and_evaluate,
)


def main():
    spark = build_spark()
    try:
        df = prepare_data(load_data(spark)).cache()
        print(f"Prepared ML rows: {df.count():,}")

        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        train_df = train_df.cache()
        test_df = test_df.cache()
        print(f"Train rows: {train_df.count():,} | Test rows: {test_df.count():,}")

        results, _ = train_and_evaluate(train_df, test_df)
        print_results_table(results)

        train_df.unpersist()
        test_df.unpersist()
        df.unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
