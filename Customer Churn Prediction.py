# Databricks notebook source
file_path = "/FileStore/tables/customer_churn_data.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)
display(df)

# COMMAND ----------

row_count = df.count()
print(f"The DataFrame contains {row_count} rows.")


# COMMAND ----------

from pyspark.sql.functions import col, mean, when, count
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
df = df.na.fill({"age": df.select(mean("age")).first()[0], "income": 0})
df = df.dropDuplicates()
display(df)


# COMMAND ----------

row_count = df.count()
print(f"The DataFrame contains {row_count} rows.")


# COMMAND ----------

from pyspark.sql.functions import datediff, current_date, max, count, sum
recency_df = df.groupBy("customer_id").agg(
    max("purchase_date").alias("last_purchase")
).withColumn("recency", datediff(current_date(), col("last_purchase")))
frequency_df = df.groupBy("customer_id").agg(count("purchase_id").alias("frequency"))

monetary_df = df.groupBy("customer_id").agg(sum("amount").alias("monetary"))

features_df = recency_df.join(frequency_df, "customer_id").join(monetary_df, "customer_id")

features_df = features_df.join(df.select("customer_id", "churn").distinct(), "customer_id")
display(features_df)


# COMMAND ----------

import matplotlib.pyplot as plt

features_pandas = features_df.toPandas()

plt.hist(features_pandas['recency'], bins=20, color='green', alpha=0.7, edgecolor='black')
plt.title("Recency Distribution")
plt.xlabel("Days Since Last Purchase")
plt.ylabel("Count")
plt.show()


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

assembler = VectorAssembler(inputCols=["recency", "frequency", "monetary"], outputCol="features")
model_data = assembler.transform(features_df).select("features", "churn").withColumnRenamed("churn", "label")

train, test = model_data.randomSplit([0.8, 0.2], seed=42)


lr = LogisticRegression(featuresCol="features", labelCol="label")


paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1, 1.0]).build()

cv = CrossValidator(
    estimator=lr,
    estimatorParamMaps=paramGrid,
    evaluator=BinaryClassificationEvaluator(),
    numFolds=3
)

model = cv.fit(train)

predictions = model.transform(test)
evaluator = BinaryClassificationEvaluator(labelCol="label")
auc = evaluator.evaluate(predictions)
print(f"Test AUC: {auc}")


# COMMAND ----------

import mlflow
import mlflow.spark

mlflow.spark.log_model(model.bestModel, "customer-churn-predictor")

print("Model saved successfully!")


# COMMAND ----------

prediction_df = predictions.select("features", "label", "prediction")

display(prediction_df)
