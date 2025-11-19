import argparse
import yaml
from google.cloud import aiplatform
from datetime import datetime


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(
    config_path: str,
    project_id: str,
    region: str,
    pipeline_root: str,
    service_account: str,
    override_version: int = None
):
    """
    Submit pipeline to Vertex AI.
    
    Args:
        config_path: Path to YAML config
        project_id: GCP project ID
        region: GCP region
        pipeline_root: GCS path for pipeline artifacts
        service_account: Service account email
        override_version: Override version from config
    """
    
    # Load config
    cfg = load_config(config_path)
    
    # Version handling
    version = override_version if override_version else int(cfg.get("version", 0)) + 1
    
    # Update config with new version
    cfg["version"] = version
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    
    print(f"Running pipeline for version {version}")
    
    # Initialize Vertex AI
    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=pipeline_root
    )
    
    # Prepare parameters
    params = {
        "config_path": config_path,
        "data_path": cfg["data_path"],
        "gcs_model_bucket": cfg["gcs_model_bucket"],
        "model_name": cfg["model_name"],
        "base_model": cfg["base_model"],
        "num_labels": int(cfg["num_labels"]),
        "learning_rate": float(cfg.get("learning_rate", 2e-5)),
        "batch_size": int(cfg.get("batch_size", 4)),
        "num_epochs": int(cfg.get("num_epochs", 8)),
        "version": version,
        "pass_threshold": float(cfg.get("pass_threshold", 0.7)),
        "bias_threshold": float(cfg.get("bias_threshold", 0.1)),
        "project_id": project_id,
        "region": region,
    }
    
    print(f"\nPipeline Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # Create pipeline job
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"{cfg['model_name']}-v{version}-{timestamp}"
    
    job = aiplatform.PipelineJob(
        display_name=job_name,
        template_path="slm_vertex_pipeline.json",
        pipeline_root=pipeline_root,
        parameter_values=params,
        enable_caching=False,
    )
    
    print(f"\nSubmitting pipeline job: {job_name}")
    print(f"Pipeline root: {pipeline_root}")
    
    # Submit
    job.submit(service_account=service_account)
    
    print(f"\n✅ Pipeline submitted successfully!")
    print(f"Job resource name: {job.resource_name}")
    print(f"\nMonitor progress:")
    print(f"  Console: https://console.cloud.google.com/vertex-ai/pipelines/runs?project={project_id}")
    print(f"  Direct link: https://console.cloud.google.com/vertex-ai/locations/{region}/pipelines/runs/{job.name.split('/')[-1]}?project={project_id}")
    
    return job


def main():
    parser = argparse.ArgumentParser(
        description="Run text classification training pipeline on Vertex AI"
    )
    
    parser.add_argument(
        "--config",
        default="configs/slm_filter.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--project-id",
        required=True,
        help="GCP project ID"
    )
    parser.add_argument(
        "--region",
        default="us-central1",
        help="GCP region"
    )
    parser.add_argument(
        "--pipeline-root",
        required=True,
        help="GCS path for pipeline artifacts (e.g., gs://bucket/pipeline_root)"
    )
    parser.add_argument(
        "--service-account",
        required=True,
        help="Service account email for pipeline execution"
    )
    parser.add_argument(
        "--version",
        type=int,
        help="Override model version (otherwise auto-incremented from config)"
    )
    
    args = parser.parse_args()
    
    try:
        run_pipeline(
            config_path=args.config,
            project_id=args.project_id,
            region=args.region,
            pipeline_root=args.pipeline_root,
            service_account=args.service_account,
            override_version=args.version
        )
    except Exception as e:
        print(f"\n❌ Pipeline submission failed: {e}")
        raise


if __name__ == "__main__":
    main()