import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast, col, mean, when

def main():
    # Initialize Serverless Spark Session
    spark = SparkSession.builder \
        .appName("FraudDetection_Distributed_ETL") \
        .getOrCreate()

    # Define GCS paths
    RAW_BUCKET = "gs://fraud-ml-data-lake/raw"
    PROCESSED_BUCKET = "gs://fraud-ml-data-lake/processed"

    print("Reading raw CSV files from GCS...")
    df_transaction = spark.read.csv(f"{RAW_BUCKET}/train_transaction.csv", header=True, inferSchema=True)
    df_identity = spark.read.csv(f"{RAW_BUCKET}/train_identity.csv", header=True, inferSchema=True)

    print("Executing Broadcast Join...")
    # Broadcast the smaller identity table to all worker nodes to prevent expensive network shuffles
    df_joined = df_transaction.join(
        broadcast(df_identity),
        on="TransactionID",
        how="left"
    )

    print("Performing Distributed Feature Engineering...")
    # 1. Handle critical missing values
    df_joined = df_joined.fillna({"card4": "UNKNOWN", "card6": "UNKNOWN"})

    # 2. Compute a rolling/group feature: Transaction amount relative to the card's average
    card_means = df_joined.groupBy("card1").agg(mean("TransactionAmt").alias("mean_card1_amt"))
    df_engineered = df_joined.join(card_means, on="card1", how="left")
    
    # Create the ratio feature
    df_engineered = df_engineered.withColumn(
        "TransactionAmt_to_Card1Mean_Ratio",
        col("TransactionAmt") / col("mean_card1_amt")
    )

    print("Writing optimized Parquet files to GCS...")
    # Repartition and write as compressed Parquet for high-speed ML training
    df_engineered.repartition(20) \
        .write \
        .mode("overwrite") \
        .parquet(f"{PROCESSED_BUCKET}/engineered_features/")

    print("Distributed ETL Pipeline Complete.")
    spark.stop()

if __name__ == "__main__":
    main()