import kfp
from kfp import dsl
from kfp.dsl import component, Output, Model, Metrics, ClassificationMetrics

@component(
    base_image="python:3.10",
    packages_to_install=["pyarrow", "fastparquet", "xgboost", "scikit-learn", "fsspec", "gcsfs"]
)
def train_fraud_model(
    data_path: str,
    model_output: Output[Model],
    metrics_output: Output[Metrics]
):
    import os
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    print(f"Reading engineered Parquet files from {data_path}...")
    df = pd.read_parquet(data_path)

    # Select numerical features and target
    target_col = "isFraud"
    ignore_cols = ["TransactionID", "isFraud", "mean_card1_amt"]
    feature_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in ignore_cols]

    X = df[feature_cols].fillna(-999)
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calculate scale_pos_weight to counter extreme fraud imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = neg_count / max(pos_count, 1) 

    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        eval_metric="auc",
        tree_method="hist",
        random_state=42
    )

    model.fit(
        X_train.values, y_train,            # Added .values here
        eval_set=[(X_val.values, y_val)],   # Added .values here
        verbose=False
    )

    os.makedirs(model_output.path, exist_ok=True)
    model_file = os.path.join(model_output.path, "model.bst")
    model.save_model(model_file)

    metrics_output.log_metric("features_used", len(feature_cols))
    metrics_output.log_metric("train_records", len(X_train))
    metrics_output.log_metric("val_records", len(X_val))


@component(
    base_image="python:3.10",
    packages_to_install=["pyarrow", "fastparquet", "xgboost", "scikit-learn", "fsspec", "gcsfs"]
)
def evaluate_fraud_model(
    data_path: str,
    model_input: kfp.dsl.Input[Model],
    metrics_output: Output[ClassificationMetrics]
) -> float:
    import os
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, confusion_matrix

    df = pd.read_parquet(data_path)
    target_col = "isFraud"
    ignore_cols = ["TransactionID", "isFraud", "mean_card1_amt"]
    feature_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in ignore_cols]

    X = df[feature_cols].fillna(-999)
    y = df[target_col]

    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier()
    model.load_model(os.path.join(model_input.path, "model.bst"))

    preds_proba = model.predict_proba(X_val.values)[:, 1] # Added .values here
    preds_binary = (preds_proba >= 0.5).astype(int)

    auc_score = float(roc_auc_score(y_val, preds_proba))
    print(f"Validation ROC-AUC: {auc_score:.4f}")

    cm = confusion_matrix(y_val, preds_binary)
    metrics_output.log_confusion_matrix(
        categories=["Legit", "Fraud"],
        matrix=cm.tolist()
    )

    return auc_score


@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform"]
)
def deploy_model_to_vertex(
    project_id: str,
    region: str,
    model_input: kfp.dsl.Input[Model],
    endpoint_name: str
):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    print("Registering model in Vertex AI Model Registry...")
    uploaded_model = aiplatform.Model.upload(
        display_name="fraud-detection-xgboost",
        artifact_uri=model_input.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest"
    )

    print("Deploying model to Vertex AI Endpoint...")
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
    uploaded_model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1
    )
    print(f"Model deployed successfully to endpoint: {endpoint.resource_name}")


# Pipeline Definition
@dsl.pipeline(
    name="continuous-fraud-detection-pipeline",
    description="End-to-end continuous training and deployment pipeline for fraud detection."
)
def fraud_pipeline(
    data_path: str = "gs://fraud-ml-data-lake/processed/engineered_features/",
    project_id: str = "stable-balancer-510511-b9", # Replace with your actual GCP Project ID
    region: str = "us-central1",
    auc_threshold: float = 0.80
):
    train_task = train_fraud_model(data_path=data_path)
    
    eval_task = evaluate_fraud_model(
        data_path=data_path,
        model_input=train_task.outputs["model_output"]
    )

    # Gate: Deploy only if performance criteria are satisfied
    with dsl.If(eval_task.outputs["Output"] >= auc_threshold, name="validation-gate"):
        deploy_model_to_vertex(
            project_id=project_id,
            region=region,
            model_input=train_task.outputs["model_output"],
            endpoint_name="fraud-detection-live-endpoint"
        )



if __name__ == "__main__":
    from kfp import compiler
    from google.cloud import aiplatform

    PROJECT_ID = "stable-balancer-510511-b9"  # Replace with your actual GCP Project ID
    REGION = "us-central1"
    PIPELINE_ROOT = "gs://fraud-ml-data-lake/pipeline_root"

    # Compile the pipeline definition into a JSON specification
    compiler.Compiler().compile(
        pipeline_func=fraud_pipeline,
        package_path="fraud_pipeline.json"
    )
    print("Compiled pipeline to fraud_pipeline.json")

    # Submit the pipeline run to Vertex AI
    aiplatform.init(project=PROJECT_ID, location=REGION)
    job = aiplatform.PipelineJob(
        display_name="fraud-detection-continuous-run-001",
        template_path="fraud_pipeline.json",
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "data_path": "gs://fraud-ml-data-lake/processed/engineered_features/",
            "project_id": PROJECT_ID,
            "region": REGION,
            "auc_threshold": 0.80
        },
        enable_caching=True
    )

    job.submit()
    print("Pipeline submitted to Vertex AI Pipelines.")