#!/usr/bin/env python
"""Build (and optionally apply) an Azure ML managed online endpoint spec.

Dry-run by default; the rendered manifest is written to
``artifacts/deploy/azureml.json``. Pass ``--apply`` with Azure credentials and
the ``azure-ai-ml`` package installed to create the endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def build_spec(
    image_uri: str,
    endpoint_name: str,
    instance_type: str,
    instance_count: int,
    subscription_id: str,
    resource_group: str,
    workspace: str,
) -> dict[str, Any]:
    """Build a managed online endpoint + deployment definition."""
    environment_variables = {
        "VECTOR_BACKEND": os.environ.get("VECTOR_BACKEND", "pgvector"),
        "POSTGRES_DSN": os.environ.get("POSTGRES_DSN", ""),
        "ARTIFACT_DIR": "/app/artifacts",
        "LLM_BACKEND": os.environ.get("LLM_BACKEND", "extractive"),
        "SEED": "42",
    }
    endpoint = {
        "name": endpoint_name,
        "type": "managed",
        "auth_mode": "key",
        "tags": {"project": "clinicalnoterag", "data": "synthetic"},
    }
    deployment = {
        "name": "blue",
        "endpoint_name": endpoint_name,
        "environment": {
            "image": image_uri,
            "inference_config": {
                "scoring_uri": "/ask",
                "scoring_port": 8000,
                "request_timeout_ms": 60000,
            },
            "environment_variables": environment_variables,
        },
        "instance_type": instance_type,
        "instance_count": instance_count,
        "scale_settings": {"type": "default"},
    }
    return {
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "workspace": workspace,
        "endpoint": endpoint,
        "deployment": deployment,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=os.environ.get("AZUREML_IMAGE_URI", ""))
    parser.add_argument("--name", default="clinicalnoterag-endpoint")
    parser.add_argument("--instance-type", default="Standard_DS3_v2")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--subscription-id", default=os.environ.get("AZURE_SUBSCRIPTION_ID", ""))
    parser.add_argument("--resource-group", default=os.environ.get("AZUREML_RESOURCE_GROUP", ""))
    parser.add_argument("--workspace", default=os.environ.get("AZUREML_WORKSPACE_NAME", ""))
    parser.add_argument("--out", default="artifacts/deploy/azureml.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True, help="Render only (default)")
    mode.add_argument("--apply", action="store_true", help="Actually create Azure resources")
    args = parser.parse_args()

    image = args.image or "myregistry.azurecr.io/clinicalnoterag:latest"
    spec = build_spec(
        image,
        args.name,
        args.instance_type,
        args.instance_count,
        args.subscription_id,
        args.resource_group,
        args.workspace,
    )
    try:
        import azure.ai.ml  # noqa: F401

        sdk_available = True
    except ImportError:
        sdk_available = False

    manifest = {
        "cloud": "azure-ml-managed-online-endpoint",
        "generated_at": datetime.now(UTC).isoformat(),
        "dry_run": not args.apply,
        "azure_ai_ml_available": sdk_available,
        "required_env": [
            "AZURE_SUBSCRIPTION_ID",
            "AZUREML_RESOURCE_GROUP",
            "AZUREML_WORKSPACE_NAME",
            "AZUREML_IMAGE_URI",
            "AZURE_CLIENT_ID / AZURE_CLIENT_SECRET / AZURE_TENANT_ID (or managed identity)",
            "POSTGRES_DSN (reachable from the endpoint's VNet or public endpoint)",
        ],
        "required_roles": [
            "AzureML Data Scientist (or a custom role with endpoint create/update)",
            "AcrPull on the container registry",
            "Key Vault Secrets User for the Postgres connection string",
        ],
        "spec": spec,
        "notes": (
            "The deployment routes /ask to the FastAPI scoring container. Add a "
            "health probe for /healthz in production. No Azure resources are created "
            "in dry-run mode."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"dry_run": manifest["dry_run"], "artifact": str(out)}, indent=2))

    if not args.apply:
        return 0
    if not sdk_available:
        raise SystemExit(
            "azure-ai-ml is required for --apply; install it in the deployment environment"
        )
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint
    from azure.identity import DefaultAzureCredential

    if not (args.subscription_id and args.resource_group and args.workspace):
        raise SystemExit("subscription id, resource group and workspace are required for --apply")
    client = MLClient(
        DefaultAzureCredential(),
        args.subscription_id,
        args.resource_group,
        args.workspace,
    )
    endpoint = ManagedOnlineEndpoint(
        name=args.name, auth_mode="key", tags={"project": "clinicalnoterag", "data": "synthetic"}
    )
    client.online_endpoints.begin_create_or_update(endpoint).result()
    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=args.name,
        image=image,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        environment_variables=spec["deployment"]["environment"]["environment_variables"],
    )
    client.online_endpoints.begin_create_or_update(deployment).result()
    print(json.dumps({"created_endpoint": args.name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
