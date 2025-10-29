# originhub-ml

Machine learning repository for OriginHub. Contains training code, configuration, and example data for the SLM filter text classification model.

## Contents

- `configs/` — YAML configuration files (e.g. `configs/slm_filter.yaml`).
- `data/` — sample and DVC-tracked data (`data/slm_filter/labeled_data.csv`).
- `models/` — model artifacts are written here (e.g. `models/slm_filter/v1`).
- `src/slm_filter/` — training code for the SLM filter model (`train.py`).
- `requirements.txt` — required Python packages.
- `gcp_credentials.json` — optional Google Cloud credentials (not committed in production).

## Quick start

1. Create and activate a Python virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Use DVC to pull data:

```bash
dvc pull
```

3. Ensure GCP credentials are available if you plan to upload models to Google Cloud Storage. By default the training script looks for `gcp_credentials.json` in the repo root, but you can set the environment variable explicitly:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/gcp_credentials.json
```

4. Run the SLM filter training pipeline (default config path is `configs/slm_filter.yaml`):

```bash
python src/slm_filter/train.py --config configs/slm_filter.yaml --log-level INFO
```

The training pipeline will:

- load and clean data from the path specified in the config
- tokenize inputs and fine-tune the model
- write the best model and `metrics.json` under `models/slm_filter/v{version}`
- upload artifacts and metrics history to the configured GCS bucket

## Configuration

See `configs/slm_filter.yaml` for default values. Important keys:

- `base_model` — HuggingFace model id used as the backbone (e.g. `distilbert-base-uncased`).
- `data_path` — CSV file with training data (expected columns: `title`, `description`, `label`).
- `output_dir` — local directory root where model versions will be saved.
- `gcs_model_bucket` — GCS bucket name used for uploading model artifacts and metrics.
- `model_name` — logical name used as the root folder in the GCS bucket.
- `num_labels` — number of classification labels.
- `batch_size`, `learning_rate`, `num_epochs`, `version` — training hyperparameters and starting version.

## Data format

The training script expects a CSV with at least these columns:

- `title` (string)
- `description` (string)
- `label` (integer or categorical mapped to integers)

Rows with missing or empty `title`/`description` are dropped. The script also removes exact duplicates.

## Where outputs are written

By default outputs are written to `models/slm_filter/v{version}`; the `metrics.json` for the run is saved there and a model metrics history file is uploaded to GCS at `{model_name}/model_metrics_history.json`.

## Troubleshooting

- If you see authentication errors when uploading to GCS, verify `GOOGLE_APPLICATION_CREDENTIALS` points to a valid service account JSON with the proper permissions.
- If data loading fails, confirm `configs/slm_filter.yaml` points to an existing CSV at `data/slm_filter/labeled_data.csv`.
- For GPU training, ensure `transformers` and supporting libraries are installed with CUDA support and that your environment offers GPU runtime.

For model-specific details and examples see `src/slm_filter/README.md`.
