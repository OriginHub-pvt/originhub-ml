# SLM Filter model

This folder contains the training pipeline for the SLM filter text classification model.

## Purpose / Contract

- Input: CSV file (configured via `configs/slm_filter.yaml`) with columns: `title`, `description`, `label`.
- Output: Saved model weights and run metrics under `models/slm_filter/v{version}`; metrics history uploaded to the configured GCS bucket under `{model_name}/model_metrics_history.json`.

The model is a Transformer-based classifier (configurable `base_model` from Hugging Face).

## How it works

`train.py` performs the following steps:

1. Loads YAML config (default `configs/slm_filter.yaml`).
2. Initializes tokenizer and training artifacts, increments the `version` in the config and prepares `models/slm_filter/v{version}`.
3. Loads and cleans data (drops rows with missing title/description, removes duplicates).
4. Tokenizes data and fine-tunes a sequence classification model using `transformers.Trainer`.
5. Saves the best model and `metrics.json` for the run.
6. Uploads artifacts and a running metrics history to Google Cloud Storage (if configured).

## Config (example keys)

See `configs/slm_filter.yaml`. Important keys and what they mean:

- `base_model` — HuggingFace model id used as backbone (string).
- `data_path` — relative path to CSV input file (string).
- `output_dir` — local directory to save model versions (string).
- `gcs_model_bucket` — GCS bucket name for uploads (string).
- `model_name` — path prefix inside GCS bucket where model artifacts are stored.
- `num_labels` — number of classes (int).
- `batch_size`, `learning_rate`, `num_epochs` — training hyperparameters.
- `version` — integer start version; `train.py` increments this value and writes it back to the YAML.

## Run examples

Basic run (uses `configs/slm_filter.yaml`):

```bash
python src/slm_filter/train.py --config configs/slm_filter.yaml --log-level INFO
```

If you want to use a different config file:

```bash
python src/slm_filter/train.py --config path/to/your_config.yaml --log-level DEBUG
```

Notes:

- The script sets `GOOGLE_APPLICATION_CREDENTIALS` to `gcp_credentials.json` by default (if not already set). You can override it before running the script.
- The script will increment `version` in the provided YAML. Commit/pin version changes as needed for reproducibility.

## Expected artifacts

- `models/slm_filter/v{version}/` — folder containing the saved model and tokenizers.
- `models/slm_filter/v{version}/metrics.json` — run metrics produced by `Trainer`.

After successful run the script uploads artifacts to GCS under `{gcs_model_bucket}/{model_name}/v{version}/`.

## Edge cases & tips

- Ensure labels are balanced enough for stratified split; the training uses stratified `train_test_split`.
- If the CSV is large, increase `batch_size` or use gradient accumulation.
- If GCS upload fails in environments without network access, disable or remove `gcs_model_bucket` from config or run in an environment with GCP credentials.

## Troubleshooting

- Authentication errors during upload: verify `GOOGLE_APPLICATION_CREDENTIALS` points to a service account JSON with write access to the target GCS bucket.
- Failing to load `base_model`: ensure internet access or that `base_model` is available locally.
- Out-of-memory errors: reduce `batch_size` or enable mixed precision training if supported.
