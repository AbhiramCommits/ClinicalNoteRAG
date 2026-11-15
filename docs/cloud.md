# Cloud deployment notes

ClinicalNoteRAG ships a container-ready FastAPI service and **dry-run**
deployment builders for AWS SageMaker and Azure ML. No cloud resources are
created by the build; the scripts write the exact request payloads to
`artifacts/deploy/` so they can be reviewed before anyone with credentials
applies them.

## Container contract

- Image: built from the repository `Dockerfile` (multi-stage, non-root UID 10001).
- Port: `8000`.
- Health: `GET /healthz`.
- Query: `POST /ask` with `{"question": "...", "filters": {...}, "k": 5}`.
- Claims: `POST /ask/claims-denial` with `{"claim_id": "CLM-0001-01"}`.
- Metrics: `GET /metrics` (reads real JSONL telemetry).
- Environment: `VECTOR_BACKEND`, `POSTGRES_DSN`, `ARTIFACT_DIR`, `LLM_BACKEND`,
  `SEED=42`. Never bake credentials into the image.

The production deployment must reach a Postgres instance with the `vector`
extension and a populated `chunks` table. Build the index with:

```bash
clinicalnoterag ingest --corpus data/corpus --store pgvector
```

## AWS SageMaker

Render the spec:

```bash
python scripts/deploy/sagemaker_deploy.py --dry-run
```

Required environment variables for a real deployment:

| Variable | Purpose |
| --- | --- |
| `AWS_REGION` | SageMaker region |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` or `AWS_PROFILE` | credentials; prefer an execution role in production |
| `SAGEMAKER_EXECUTION_ROLE_ARN` | role assumed by the model container to pull ECR and write logs |
| `ECR_IMAGE_URI` | full image URI including tag/digest |
| `POSTGRES_DSN` | reachable pgvector connection string (use Secrets Manager in production) |

Minimum IAM actions: `sagemaker:CreateModel`, `sagemaker:CreateEndpointConfig`,
`sagemaker:CreateEndpoint`, `sagemaker:DescribeEndpoint`,
`ecr:GetAuthorizationToken`, `ecr:BatchGetImage`.

The FastAPI container exposes `/ask`, while SageMaker's default invocation path
is `/invocations`. For a production endpoint, add a small ASGI proxy or set the
container command to a SageMaker handler that forwards `/invocations` to the
answerer. The dry-run manifest records this as a required integration step.

## Azure ML

Render the spec:

```bash
python scripts/deploy/azureml_deploy.py --dry-run
```

Required environment variables:

| Variable | Purpose |
| --- | --- |
| `AZURE_SUBSCRIPTION_ID` | target subscription |
| `AZUREML_RESOURCE_GROUP` | resource group |
| `AZUREML_WORKSPACE_NAME` | Azure ML workspace |
| `AZUREML_IMAGE_URI` | ACR image URI |
| `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID` | service principal, or use a managed identity |
| `POSTGRES_DSN` | pgvector connection string stored in Key Vault |

Required roles: AzureML Data Scientist (or a custom endpoint role), `AcrPull`
on the registry, and `Key Vault Secrets User`.

## Deployment safety

- `--dry-run` is the default documented path. `--apply` requires explicit
  credentials and creates billable resources.
- Do not put patient data in `POSTGRES_DSN` or environment variables.
- The service is a research prototype, not a validated medical device. Do not
  expose it to real PHI without a HIPAA BAA, access controls, audit logging,
  encryption review, and clinical governance sign-off.
