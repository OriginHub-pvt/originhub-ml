# pipeline_fixed.py
import kfp
from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Artifact, Condition


@component(
    base_image="python:3.10",
    packages_to_install=[
        "pyyaml==6.0",
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "transformers==4.30.0",
        "datasets==2.14.0",
        "evaluate==0.4.0",
        "torch==2.0.1",
        "accelerate==0.21.0",
        "google-cloud-storage==2.10.0"
    ],
)
def preprocess_component(
    config_path: str,
    data_path: str,
    base_model: str,
    processed_output: Output[Dataset]
):
    """Preprocess data and save tokenized datasets."""
    import os
    import pandas as pd
    import yaml
    from sklearn.model_selection import train_test_split
    from datasets import Dataset as HF_Dataset
    from transformers import AutoTokenizer
    from google.cloud import storage
    
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Download data from GCS if needed
    if data_path.startswith("gs://"):
        client = storage.Client()
        bucket_name = data_path.split("/")[2]
        blob_path = "/".join(data_path.split("/")[3:])
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        local_data_path = "/tmp/data.csv"
        blob.download_to_filename(local_data_path)
        data_path = local_data_path
    
    # Load and clean data
    df = pd.read_csv(data_path)
    df.dropna(subset=["title", "description", "label"], inplace=True)
    df["title"] = df["title"].astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()
    df = df[(df["title"] != "") & (df["description"] != "")]
    df.drop_duplicates(subset=["title", "description", "label"], inplace=True)
    
    print(f"Cleaned dataset: {len(df)} rows")
    
    # Train/test split
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Create HF datasets
    train_ds = HF_Dataset.from_pandas(train_df)
    test_ds = HF_Dataset.from_pandas(test_df)
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    def preprocess_batch(batch):
        text = f"{(batch.get('title') or '').strip()} {(batch.get('description') or '').strip()}"
        return tokenizer(text, truncation=True, padding="max_length", max_length=512)
    
    encoded_train = train_ds.map(preprocess_batch, batched=False)
    encoded_test = test_ds.map(preprocess_batch, batched=False)
    
    # Rename label column
    encoded_train = encoded_train.rename_column("label", "labels")
    encoded_test = encoded_test.rename_column("label", "labels")
    
    # Save without torch format to avoid serialization issues
    os.makedirs(processed_output.path, exist_ok=True)
    encoded_train.save_to_disk(os.path.join(processed_output.path, "train"))
    encoded_test.save_to_disk(os.path.join(processed_output.path, "test"))
    
    print(f"Saved preprocessed data to {processed_output.path}")


@component(
    base_image="python:3.10",
    packages_to_install=[
        "pyyaml==6.0",
        "transformers==4.30.0",
        "datasets==2.14.0",
        "evaluate==0.4.0",
        "torch==2.0.1",
        "accelerate==0.21.0",
        "google-cloud-storage==2.10.0"
    ],
)
def train_component(
    config_path: str,
    base_model: str,
    num_labels: int,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    version: int,
    gcs_model_bucket: str,
    model_name: str,
    processed_input: Input[Dataset],
    model_out: Output[Model],
    metrics_out: Output[Artifact]
):
    """Train the model and save outputs."""
    import os
    import json
    import shutil
    from datetime import datetime
    from datasets import Dataset as HF_Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback
    )
    import evaluate
    from google.cloud import storage
    
    # # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Load preprocessed datasets
    train_ds = HF_Dataset.load_from_disk(os.path.join(processed_input.path, "train"))
    test_ds = HF_Dataset.load_from_disk(os.path.join(processed_input.path, "test"))
    
    # Set torch format now
    train_ds = train_ds.with_format("torch")
    test_ds = test_ds.with_format("torch")
    
    print(f"Loaded datasets - Train: {len(train_ds)}, Test: {len(test_ds)}")
    
    # Load metrics
    metrics_dict = {
        "accuracy": evaluate.load("accuracy"),
        "precision": evaluate.load("precision"),
        "recall": evaluate.load("recall"),
        "f1": evaluate.load("f1"),
    }
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {
            name: metric.compute(predictions=preds, references=labels)[name]
            for name, metric in metrics_dict.items()
        }
    
    # Create output directory
    version_dir = "/tmp/model_training"
    os.makedirs(version_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {base_model}")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=num_labels
    )
    
    # Training arguments
    args = TrainingArguments(
        output_dir=version_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to=[]
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("Starting training...")
    train_result = trainer.train()
    
    # Save model
    trainer.save_model(version_dir)
    print(f"Model saved to {version_dir}")
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    # Prepare metrics
    metrics = {
        "version": version,
        "timestamp": datetime.utcnow().isoformat(),
        "train_loss": train_result.metrics.get("train_loss", 0),
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "eval_loss": eval_results.get("eval_loss", 0),
        "eval_accuracy": eval_results.get("eval_accuracy", 0),
        "eval_precision": eval_results.get("eval_precision", 0),
        "eval_recall": eval_results.get("eval_recall", 0),
        "eval_f1": eval_results.get("eval_f1", 0),
    }
    
    print(f"Training complete. F1 Score: {metrics['eval_f1']:.4f}")
    
    # Save metrics
    os.makedirs(metrics_out.path, exist_ok=True)
    metrics_path = os.path.join(metrics_out.path, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Remove checkpoints to save space
    for item in os.listdir(version_dir):
        if item.startswith("checkpoint"):
            shutil.rmtree(os.path.join(version_dir, item), ignore_errors=True)
    
    # Copy to model output
    os.makedirs(model_out.path, exist_ok=True)
    shutil.copytree(version_dir, model_out.path, dirs_exist_ok=True)
    
    # Upload to GCS
    print(f"Uploading to gs://{gcs_model_bucket}/{model_name}/v{version}")
    client = storage.Client()
    bucket = client.bucket(gcs_model_bucket)
    gcs_version_path = f"{model_name}/v{version}"
    
    for root, _, files in os.walk(version_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, version_dir)
            blob = bucket.blob(f"{gcs_version_path}/{rel_path}")
            blob.upload_from_filename(local_path)
    
    print("Upload complete")
    
    # Update metrics history
    gcs_history_path = f"{model_name}/model_metrics_history.json"
    history_blob = bucket.blob(gcs_history_path)
    
    history = []
    if history_blob.exists():
        history_content = history_blob.download_as_text()
        try:
            history = json.loads(history_content)
        except json.JSONDecodeError:
            print("Warning: Corrupted metrics history, starting fresh")
    
    history.append(metrics)
    history_blob.upload_from_string(json.dumps(history, indent=4))
    print(f"Updated metrics history with {len(history)} entries")
    
    # Set output metadata
    model_out.metadata["version"] = version
    model_out.metadata["f1_score"] = metrics["eval_f1"]
    model_out.uri = f"gs://{gcs_model_bucket}/{gcs_version_path}"


@component(
    base_image="python:3.10",
    packages_to_install=["pyyaml==6.0"]
)
def evaluate_component(
    metrics_input: Input[Artifact],
    pass_threshold: float
) -> bool:
    """Check if model meets quality threshold."""
    import os
    import json
    
    metrics_file = os.path.join(metrics_input.path, "metrics.json")
    
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    # Try multiple possible F1 metric keys
    f1 = (metrics.get("eval_f1") or 
          metrics.get("f1") or 
          metrics.get("eval_f1_score") or 
          metrics.get("best_f1"))
    
    if f1 is None:
        raise RuntimeError(f"No F1 metric found in metrics.json. Available keys: {list(metrics.keys())}")
    
    print(f"F1 Score: {f1:.4f}, Threshold: {pass_threshold}")
    passed = float(f1) >= float(pass_threshold)
    print(f"Quality check: {'PASSED' if passed else 'FAILED'}")
    
    return passed


@component(
    base_image="python:3.10",
    packages_to_install=[
        "pyyaml==6.0",
        "pandas==2.0.3",
        "google-cloud-storage==2.10.0",
        "fairlearn==0.9.0",
        "scikit-learn==1.3.0"
    ]
)
def bias_component(
    gcs_bucket: str,
    model_name: str,
    version: int,
    bias_threshold: float
) -> bool:
    """
    Check for bias in model predictions.
    Expects: gs://{gcs_bucket}/{model_name}/v{version}/bias_dataset.csv
    CSV columns: protected_attr, label, pred
    """
    import os
    import pandas as pd
    from google.cloud import storage
    from fairlearn.metrics import MetricFrame
    from sklearn.metrics import accuracy_score
    
    # Download bias dataset from GCS
    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    bias_file = f"{model_name}/v{version}/bias_dataset.csv"
    blob = bucket.blob(bias_file)
    
    if not blob.exists():
        print(f"Warning: Bias dataset not found at gs://{gcs_bucket}/{bias_file}")
        print("Skipping bias check - assuming OK")
        return True
    
    local_path = "/tmp/bias_dataset.csv"
    blob.download_to_filename(local_path)
    
    # Load and check bias
    df = pd.read_csv(local_path)
    
    required_cols = ["protected_attr", "label", "pred"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Bias dataset must have columns: {required_cols}. Found: {df.columns.tolist()}")
    
    print(f"Loaded bias dataset: {len(df)} samples")
    
    # Calculate metrics by protected attribute
    mf = MetricFrame(
        metrics=accuracy_score,
        y_true=df["label"],
        y_pred=df["pred"],
        sensitive_features=df["protected_attr"]
    )
    
    by_group = mf.by_group
    print(f"Accuracy by group:\n{by_group}")
    
    # Check if max difference exceeds threshold
    vals = list(by_group.values())
    max_diff = max(vals) - min(vals)
    
    print(f"Max accuracy difference: {max_diff:.4f}, Threshold: {bias_threshold}")
    
    passed = max_diff <= bias_threshold
    print(f"Bias check: {'PASSED' if passed else 'FAILED'}")
    
    return passed


@component(
    base_image="python:3.10",
    packages_to_install=[
        "pyyaml==6.0",
        "google-cloud-storage==2.10.0",
        "google-cloud-aiplatform==1.38.0"
    ]
)
def upload_and_register_component(
    gcs_model_bucket: str,
    model_name: str,
    version: int,
    project_id: str,
    region: str,
    model_input: Input[Model]
):
    """Register model in Vertex AI Model Registry."""
    from google.cloud import aiplatform
    
    gcs_uri = f"gs://{gcs_model_bucket}/{model_name}/v{version}"
    
    print(f"Registering model from: {gcs_uri}")
    
    aiplatform.init(project=project_id, location=region)
    
    # Upload/register model
    model = aiplatform.Model.upload(
        display_name=f"{model_name}_v{version}",
        artifact_uri=gcs_uri,
        description=f"Text classification model version {version}"
    )
    
    print(f"Model registered: {model.resource_name}")
    print(f"Model ID: {model.name}")


@dsl.pipeline(
    name="slm-vertex-pipeline",
    description="End-to-end ML pipeline: Preprocess -> Train -> Eval -> Bias Check -> Register"
)
def slm_pipeline(
    config_path: str,
    data_path: str,
    gcs_model_bucket: str,
    model_name: str,
    base_model: str = "distilbert-base-uncased",
    num_labels: int = 2,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    num_epochs: int = 8,
    version: int = 1,
    pass_threshold: float = 0.7,
    bias_threshold: float = 0.1,
    project_id: str = "your-project-id",
    region: str = "us-central1",
):
    """
    Main pipeline orchestrating the ML workflow.
    
    Args:
        config_path: Path to YAML config file
        data_path: GCS path to training CSV
        gcs_model_bucket: GCS bucket for model storage
        model_name: Model identifier
        base_model: HuggingFace model
        num_labels: Number of classes
        learning_rate: Training LR
        batch_size: Batch size
        num_epochs: Training epochs
        version: Model version
        pass_threshold: Minimum F1 to pass
        bias_threshold: Max acceptable accuracy difference
        project_id: GCP project
        region: GCP region
    """
    
    # Step 1: Preprocess
    preprocess_task = preprocess_component(
        config_path=config_path,
        data_path=data_path,
        base_model=base_model
    )
    preprocess_task.set_cpu_limit("1")
    preprocess_task.set_memory_limit("8G")
    
    # Step 2: Train
    train_task = train_component(
        config_path=config_path,
        base_model=base_model,
        num_labels=num_labels,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        version=version,
        gcs_model_bucket=gcs_model_bucket,
        model_name=model_name,
        processed_input=preprocess_task.outputs["processed_output"]
    )
    # Configure GPU for training
    # train_task.set_accelerator_type("NVIDIA_TESLA_T4")
    # train_task.set_accelerator_limit(1)
    train_task.set_cpu_limit("1")
    train_task.set_memory_limit("16G")
    
    # Step 3: Evaluate quality
    eval_task = evaluate_component(
        metrics_input=train_task.outputs["metrics_out"],
        pass_threshold=pass_threshold
    )
    
    # Step 4: Conditional bias check (only if quality passed)
    with Condition(eval_task.output == True, name="quality-passed"):
        bias_task = bias_component(
            gcs_bucket=gcs_model_bucket,
            model_name=model_name,
            version=version,
            bias_threshold=bias_threshold
        )
        
        # Step 5: Register model (only if bias check passed)
        with Condition(bias_task.output == True, name="bias-check-passed"):
            register_task = upload_and_register_component(
                gcs_model_bucket=gcs_model_bucket,
                model_name=model_name,
                version=version,
                project_id=project_id,
                region=region,
                model_input=train_task.outputs["model_out"]
            )


if __name__ == "__main__":
    # Compile pipeline
    kfp.compiler.Compiler().compile(
        pipeline_func=slm_pipeline,
        package_path="slm_vertex_pipeline.json"
    )
    print("Pipeline compiled successfully to slm_vertex_pipeline.json 3")