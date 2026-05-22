# SE446 Milestone 2 - Chicago Crime Analytics with Spark + MLlib

## Team Members

| Name | ID | Main Contribution |
|---|---:|---|
| Abdulatif Alabdulatif | 200291 | Tasks 8-10, deployment evidence |
| Bakr Jamjoom | 210084 | Tasks 1-2, Spark DataFrame and SQL analytics |
| Saleh Alhaidar | 230417 | Tasks 3-4, yearly trends and arrest-rate analysis |
| Abdullah Almutabagani | 220534 | Tasks 5-7, Spark ML pipeline and model evaluation |

## Executive Summary

This project upgrades the Chicago Crimes analysis from MapReduce-style counting to Spark DataFrame, Spark SQL, and MLlib workflows. The analytics tasks reproduce the M1 crime counts, location hotspots, year trends, and arrest-rate analysis, while the ML tasks train classifiers to predict whether a crime results in an arrest.

## Dataset

- Cluster dataset: `hdfs:///data/chicago_crimes.csv`
- Local mode: generated 10,000-row sample with the same core schema
- Main label for ML: `Arrest`
- Main ML features: `District`, `Primary Type`, `Hour`, `Domestic`

## M1 vs M2 Comparison

| Task | M1 Approach | M2 Approach | Expected Result |
|---|---|---|---|
| Task 1: Crime type distribution | Mapper, sort, reducer | Spark DataFrame `groupBy().count()` | Same top crime counts on the same dataset |
| Task 2: Location hotspots | Mapper, sort, reducer | Spark SQL query over temp view | Same top location counts on the same dataset |
| Task 3: Yearly trend | CSV parsing and counting | Spark DataFrame group by `Year` | Same yearly counts, plus local visualization |
| Task 4: Arrest rate | CSV/Pandas counting | Spark DataFrame aggregation | Same overall rate, plus per-crime-type rates |

Spark is shorter and easier to iterate on because repeated aggregations can stay in memory and do not require separate mapper/reducer scripts.

## ML Results Summary

The ML pipeline is implemented in `m2_spark_ml.py` and `src/T11M2.py`.

It performs:

- Task 5: feature engineering with `StringIndexer` and `VectorAssembler`
- Task 6: training/evaluation of Logistic Regression, Random Forest, and GBT
- Task 7: Random Forest feature importance reporting

The feature vector is:

```text
[District, crime_index, Hour, domestic_index]
```

Final model metrics should be copied from the cluster `spark-submit` output after running:

```bash
spark-submit --master yarn --deploy-mode client --num-executors 2 --executor-memory 1g --executor-cores 2 m2_spark_ml.py
```

## Deployment Evidence

### Task 9: Local Execution

Run:

```bash
spark-submit --master local[*] src/T9M2.py
```

Evidence should show:

```text
Master: local[*]
Generated row count: 10000
```

### Task 10: Cluster Client Mode

Run:

```bash
spark-submit --master yarn --deploy-mode client src/T10M2.py
```

Evidence should show:

```text
Master: yarn
Real row count: 7M+
```

### Task 11: Spark Submit

Run:

```bash
spark-submit \
  --master yarn \
  --deploy-mode client \
  --num-executors 2 \
  --executor-memory 1g \
  --executor-cores 2 \
  m2_spark_ml.py
```

The terminal output should include the feature sample, model comparison table, confusion matrices, and Random Forest feature importances.

## How to Run

Local setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run local generated-data evidence:

```bash
spark-submit --master local[*] src/T9M2.py
```

Run the full ML pipeline locally with generated data:

```bash
SPARK_MASTER='local[*]' spark-submit m2_spark_ml.py
```

Run on the cluster:

```bash
spark-submit --master yarn --deploy-mode client src/T10M2.py
spark-submit --master yarn --deploy-mode client --num-executors 2 --executor-memory 1g --executor-cores 2 m2_spark_ml.py
```

## Repository Structure

```text
.
├── M2_Spark_ML_GroupSAAB.ipynb
├── README.md
├── m2_spark_ml.py
├── requirements.txt
├── docs/
├── output/
└── src/
```

## Notes

The old M1 scripts in `src/Task*.py` are retained for comparison and evidence, but the M2 grading requirements are primarily covered by the notebook, `m2_spark_ml.py`, and the cluster `spark-submit` outputs.
