#!/usr/bin/env python
"""Build (and optionally apply) an AWS SageMaker endpoint deployment spec.

By default this script is dry-run only: it writes the exact boto3 request
payloads to ``artifacts/deploy/sagemaker.json`` without creating cloud
resources. Pass ``--apply`` and provide real AWS credentials/roles to deploy.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def build_specs(
    image_uri: str,
    role_arn: str,
    model_name: str,
    endpoint_name: str,
    instance_type: str,
    instance_count: int,
) -> dict[str, Any]:
    """Build the three SageMaker API request payloads."""
    environment = {
        "VECTOR_BACKEND": os.environ.get("VECTOR_BACKEND", "pgvector"),
        "POSTGRES_DSN": os.environ.get("POSTGRES_DSN", ""),
        "ARTIFACT_DIR": "/opt/ml/model/artifacts",
        "LLM_BACKEND": os.environ.get("LLM_BACKEND", "extractive"),
        "SEED": "42",
    }
    model = {
        "ModelName": model_name,
        "PrimaryContainer": {
            "Image": image_uri,
            "Environment": environment,
            "Mode": "SingleModel",
        },
        "ExecutionRoleArn": role_arn,
    }
    endpoint_config = {
        "EndpointConfigName": f"{endpoint_name}-config",
        "ProductionVariants": [
            {
                "VariantName": "primary",
                "ModelName": model_name,
                "InitialInstanceCount": instance_count,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    }
    endpoint = {
        "EndpointName": endpoint_name,
        "EndpointConfigName": f"{endpoint_name}-config",
    }
    return {
        "model": model,
        "endpoint_config": endpoint_config,
        "endpoint": endpoint,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=os.environ.get("ECR_IMAGE_URI", ""))
    parser.add_argument(
        "--role",
        default=os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN", ""),
        help="SageMaker execution role ARN",
    )
    parser.add_argument("--name", default="clinicalnoterag-rag")
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--out", default="artifacts/deploy/sagemaker.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True, help="Render only (default)")
    mode.add_argument("--apply", action="store_true", help="Actually create AWS resources")
    args = parser.parse_args()

    image = args.image or ("123456789012.dkr.ecr.us-east-1.amazonaws.com/clinicalnoterag:latest")
    role = args.role or "arn:aws:iam::123456789012:role/ClinicalNoteRAGExecutionRole"
    specs = build_specs(image, role, args.name, args.name, args.instance_type, args.instance_count)
    try:
        import boto3  # noqa: F401

        boto3_available = True
    except ImportError:
        boto3_available = False

    manifest = {
        "cloud": "aws-sagemaker",
        "generated_at": datetime.now(UTC).isoformat(),
        "dry_run": not args.apply,
        "boto3_available": boto3_available,
        "required_env": [
            "AWS_REGION",
            "AWS_ACCESS_KEY_ID / AWS_PROFILE (or an instance/task role)",
            "SAGEMAKER_EXECUTION_ROLE_ARN",
            "ECR_IMAGE_URI",
            "POSTGRES_DSN (reachable from the VPC or public endpoint)",
        ],
        "required_iam_actions": [
            "sagemaker:CreateModel",
            "sagemaker:CreateEndpointConfig",
            "sagemaker:CreateEndpoint",
            "sagemaker:DescribeEndpoint",
            "ecr:GetAuthorizationToken",
            "ecr:BatchGetImage",
        ],
        "image": image,
        "role_arn": role,
        "specs": specs,
        "notes": (
            "The API container listens on port 8000 and implements /healthz and /ask. "
            "SageMaker invokes the container with /invocations; add a thin proxy or "
            "set SAGEMAKER_PROGRAM=/app/clinicalnoterag/api/sagemaker_entry.py for a "
            "production endpoint."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"dry_run": manifest["dry_run"], "artifact": str(out)}, indent=2))

    if not args.apply:
        return 0
    if not boto3_available:
        raise SystemExit("boto3 is required for --apply; install it in the deployment environment")
    import boto3

    client = boto3.client("sagemaker", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    client.create_model(**specs["model"])
    client.create_endpoint_config(**specs["endpoint_config"])
    client.create_endpoint(**specs["endpoint"])
    print(json.dumps({"created_endpoint": specs["endpoint"]["EndpointName"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
