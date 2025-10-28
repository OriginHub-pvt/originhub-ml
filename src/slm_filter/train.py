import os
import json
import yaml
import shutil
import argparse
import logging
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import evaluate
from google.cloud import storage

def setup_logging(level_name: str):
    """
    Configure logging for the entire module.

    Args:
        level_name (str): Logging level as a string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level_name.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level_name}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logging.info(f"Logging initialized at level: {level_name.upper()}")


class ModelTrainer:
    """Trainer for text classification with versioning, evaluation, and GCS upload."""

    def __init__(self, config_path: str):
        try:
            logging.debug("Initializing ModelTrainer...")
            self.config_path = config_path
            self.cfg = yaml.safe_load(open(config_path))
            logging.debug(f"Loaded configuration from {config_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["base_model"])
            self.metrics = {
                "accuracy": evaluate.load("accuracy"),
                "precision": evaluate.load("precision"),
                "recall": evaluate.load("recall"),
                "f1": evaluate.load("f1"),
            }

            # Version increment and directory setup
            self.version = int(self.cfg.get("version", 0)) + 1
            self.cfg["version"] = self.version
            self.save_config()

            self.version_dir = os.path.join(self.cfg["output_dir"], f"v{self.version}")
            os.makedirs(self.version_dir, exist_ok=True)

            logging.info(f"Initialized trainer for model version v{self.version}")
        except Exception as e:
            logging.error(f"Failed to initialize ModelTrainer: {e}", exc_info=True)
            raise

    def save_config(self) -> None:
        """Save updated configuration back to YAML."""
        try:
            with open(self.config_path, "w") as f:
                yaml.dump(self.cfg, f)
            logging.debug(f"Configuration updated to version {self.cfg['version']}")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}", exc_info=True)
            raise

    def load_and_clean_data(self) -> tuple[Dataset, Dataset]:
        """Load dataset, clean text, and stratify train/test split."""
        try:
            df = pd.read_csv(self.cfg["data_path"])
            logging.debug(f"Loaded dataset from {self.cfg['data_path']}")

            df.dropna(subset=["title", "description", "label"], inplace=True)
            df["title"] = df["title"].astype(str).str.strip()
            df["description"] = df["description"].astype(str).str.strip()
            df = df[(df["title"] != "") & (df["description"] != "")]
            df.drop_duplicates(subset=["title", "description", "label"], inplace=True)

            train_df, test_df = train_test_split(
                df, test_size=0.2, stratify=df["label"], random_state=42
            )
            logging.info(f"Dataset cleaned: {len(train_df)} train, {len(test_df)} test samples")
            return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)
        except Exception as e:
            logging.error("Error loading or cleaning dataset", exc_info=True)
            raise

    def preprocess_batch(self, batch: dict) -> dict:
        """Tokenize title and description fields."""
        text = f"{(batch['title'] or '').strip()} {(batch['description'] or '').strip()}"
        return self.tokenizer(text, truncation=True, padding="max_length")

    def preprocess_dataset(self, train_ds: Dataset, test_ds: Dataset) -> tuple[Dataset, Dataset]:
        """Tokenize datasets for training and testing."""
        try:
            logging.debug("Preprocessing datasets...")
            encoded_train = train_ds.map(self.preprocess_batch)
            encoded_test = test_ds.map(self.preprocess_batch)

            encoded_train = encoded_train.rename_column("label", "labels").with_format("torch")
            encoded_test = encoded_test.rename_column("label", "labels").with_format("torch")

            logging.info("Dataset preprocessing complete.")
            return encoded_train, encoded_test
        except Exception as e:
            logging.error("Preprocessing failed", exc_info=True)
            raise

    def compute_metrics(self, eval_pred) -> dict:
        """Compute accuracy, precision, recall, and F1 metrics."""
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {
            name: metric.compute(predictions=preds, references=labels)[name]
            for name, metric in self.metrics.items()
        }

    def train_model(self, train_ds: Dataset, test_ds: Dataset) -> str:
        """Train model and save best weights."""
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.cfg["base_model"], num_labels=self.cfg["num_labels"]
            )

            args = TrainingArguments(
                output_dir=self.version_dir,
                learning_rate=float(self.cfg.get("learning_rate", 2e-5)),
                per_device_train_batch_size=int(self.cfg.get("batch_size", 4)),
                num_train_epochs=int(self.cfg.get("num_epochs", 8)),
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                save_total_limit=1,
                report_to=[]
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=test_ds,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )

            logging.info("Starting training...")
            train_result = trainer.train()
            trainer.save_model(self.version_dir)
            logging.info("Model training completed and best weights saved.")

            metrics = train_result.metrics
            metrics["version"] = self.version
            metrics["timestamp"] = datetime.utcnow().isoformat()

            metrics_path = os.path.join(self.version_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)

            logging.debug(f"Metrics written to {metrics_path}")
            self.update_metrics_history(metrics)
            return metrics_path
        except Exception as e:
            logging.error("Error during training", exc_info=True)
            raise

    def update_metrics_history(self, new_metrics: dict) -> None:
        """Append metrics to GCS-based metrics history file."""
        try:
            client = storage.Client()
            bucket = client.bucket(self.cfg["gcs_model_bucket"])
            gcs_path = f"{self.cfg['model_name']}/model_metrics_history.json"
            blob = bucket.blob(gcs_path)

            history = []
            history_path = os.path.join(self.cfg["output_dir"], "model_metrics_history.json")

            if blob.exists():
                tmp_path = os.path.join(self.cfg["output_dir"], "_tmp_history.json")
                blob.download_to_filename(tmp_path)
                try:
                    with open(tmp_path, "r") as f:
                        history = json.load(f)
                    logging.debug(f"Loaded {len(history)} previous history entries.")
                except json.JSONDecodeError:
                    logging.warning("Corrupted GCS metrics file. Starting new history.")
                os.remove(tmp_path)

            history.append(new_metrics)
            with open(history_path, "w") as f:
                json.dump(history, f, indent=4)
            blob.upload_from_filename(history_path)
            os.remove(history_path)

            logging.info(f"Uploaded updated metrics history with {len(history)} entries.")
        except Exception as e:
            logging.error("Failed to update metrics history", exc_info=True)
            raise

    def upload_to_gcs(self) -> None:
        """Upload model version to GCS, removing checkpoints."""
        try:
            client = storage.Client()
            bucket = client.bucket(self.cfg["gcs_model_bucket"])
            gcs_version_path = os.path.join(self.cfg["model_name"], f"v{self.version}")

            for root, dirs, _ in os.walk(self.version_dir):
                for d in dirs:
                    if d.startswith("checkpoint"):
                        shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                        logging.debug(f"Removed checkpoint folder: {os.path.join(root, d)}")

            for root, _, files in os.walk(self.version_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    rel_path = os.path.relpath(local_path, self.version_dir)
                    blob = bucket.blob(f"{gcs_version_path}/{rel_path}")
                    blob.upload_from_filename(local_path)
                    logging.debug(f"Uploaded {rel_path} to GCS at {gcs_version_path}/")

            models_root = self.cfg["output_dir"]
            shutil.rmtree(models_root, ignore_errors=True)
            logging.info(f"Removed entire local models folder: {models_root}")

        except Exception as e:
            logging.error("Error during GCS upload", exc_info=True)
            raise

def main(config_path: str, log_level: str) -> None:
    """Main entrypoint for model training pipeline."""
    setup_logging(log_level)

    try:
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "gcp_credentials.json")

        trainer = ModelTrainer(config_path)
        train_ds, test_ds = trainer.load_and_clean_data()
        encoded_train, encoded_test = trainer.preprocess_dataset(train_ds, test_ds)
        trainer.train_model(encoded_train, encoded_test)
        trainer.upload_to_gcs()

        logging.info("Pipeline execution completed successfully.")
    except Exception as e:
        logging.critical("Pipeline execution failed", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and upload text classification model.")
    parser.add_argument("--config", default="configs/slm_filter.yaml", help="Path to YAML config file.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    args = parser.parse_args()

    main(args.config, args.log_level)
