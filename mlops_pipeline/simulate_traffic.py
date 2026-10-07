import time
import uuid
import pandas as pd
from datetime import datetime
from google.cloud import aiplatform, bigquery

# Initialize clients
PROJECT_ID = "stable-balancer-510511-b9"  # Replace with your actual GCP Project ID
REGION = "us-central1"

aiplatform.init(project=PROJECT_ID, location=REGION)
bq_client = bigquery.Client(project=PROJECT_ID)

print("Fetching active Vertex AI Endpoint...")
endpoints = aiplatform.Endpoint.list(filter='display_name="fraud-detection-live-endpoint"')
if not endpoints:
    raise ValueError("Endpoint not found. Did the pipeline finish deploying?")
endpoint = endpoints[0]

print("Loading sample validation data...")
df = pd.read_parquet("gs://fraud-ml-data-lake/processed/engineered_features/").head(100)

ignore_cols = ["TransactionID", "isFraud", "mean_card1_amt"]
feature_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in ignore_cols]

# Fill nulls with the same -999 value used during training
X_sample = df[feature_cols].fillna(-999).values.tolist()

print("Simulating live traffic and logging to BigQuery...")
table_id = f"{PROJECT_ID}.fraud_lake_dw.endpoint_logs"

for i, instance in enumerate(X_sample):
    # 1. Get prediction from Vertex AI
    prediction = endpoint.predict(instances=[instance])
    
    # Vertex AI XGBoost binary classifier returns a single float for the positive class
    fraud_prob = prediction.predictions[0] 
    is_fraud_flag = 1 if fraud_prob >= 0.50 else 0
    
    # 2. Prepare BigQuery log
    row_to_insert = [{
        "timestamp": datetime.utcnow().isoformat(),
        "transaction_id": str(uuid.uuid4()),
        "predicted_fraud_probability": float(fraud_prob),
        "is_fraud_flag": is_fraud_flag
    }]
    
    # 3. Stream to BigQuery
    errors = bq_client.insert_rows_json(table_id, row_to_insert)
    if not errors:
        print(f"Transaction {i+1} logged. Fraud Prob: {fraud_prob:.4f} | Flag: {is_fraud_flag}")
    
    time.sleep(1) # Simulate real-time stream delay

print("Traffic simulation complete!")