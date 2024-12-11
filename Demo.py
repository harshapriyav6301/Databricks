# Databricks notebook source
file_path = "/FileStore/tables/retail_data.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)
display(df)

# COMMAND ----------

row_count = df.count()
row_count

# COMMAND ----------

df_clean = df.dropna()
df_clean = df_clean.dropDuplicates(["Transaction_ID"])
display(df_clean)

# COMMAND ----------

row_count = df_clean.count()
row_count

# COMMAND ----------

from pyspark.sql.functions import round
df_clean = df_clean.withColumn("Amount", round(df_clean["Amount"], 2))
df_clean = df_clean.withColumn("Total_Amount", round(df_clean["Total_Amount"], 2))
display(df_clean)


# COMMAND ----------

from pyspark.sql.functions import to_date, regexp_replace, when, col, date_format
df_transformed = df_clean.withColumn("Date", regexp_replace("Date", "/", "-"))
df_transformed = df_transformed.withColumn(
"Date",
when(
col("Date").rlike("^[0-9]{1,2}-[0-9]{1,2}-[0-9]{4}$"),
date_format(to_date("Date", "M-d-yyyy"), "MM-dd-yyyy")
)
.when(
col("Date").rlike("^[0-9]{2}-[0-9]{2}-[0-9]{2}$"),
date_format(to_date("Date", "MM-dd-yy"), "MM-dd-yyyy")
)
.otherwise(None)
)
df_transformed= df_transformed.withColumn("Date", to_date("Date","MM-dd-yyyy"))
display(df_transformed)


# COMMAND ----------

storage_account_name = "storagedemo63"
container_name = "demo"
sas_token = "sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2024-12-17T22:34:47Z&st=2024-12-10T14:34:47Z&spr=https&sig=OPZGCYNu7FQGMUJ9cEbDO65tk4JOVxTnvo%2FTZiOqr7A%3D"
dbutils.fs.mount(
source=f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/",
mount_point="/mnt/mycontainer1",
extra_configs={f"fs.azure.sas.{container_name}.{storage_account_name}.blob.core.windows.net": 
sas_token}
)


# COMMAND ----------

output_path = "/mnt/mycontainer1/processed_data.csv"
df_transformed.coalesce(1) \
.write.format("csv") \
.option("header", "true") \
.save(output_path)