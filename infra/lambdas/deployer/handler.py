"""Deploy an approved model package to the Serverless Inference endpoint.

Triggered by EventBridge: "SageMaker Model Package State Change" with
ModelApprovalStatus = Approved, for the query-sentinel model package group.
UpdateEndpoint is blue/green, so traffic moves only once the new variant is healthy.
"""
from __future__ import annotations

import os
import time

import boto3
from botocore.exceptions import ClientError

sm = boto3.client("sagemaker")

ENDPOINT = os.environ["ENDPOINT_NAME"]
ROLE_ARN = os.environ["MODEL_ROLE_ARN"]
MEMORY_MB = int(os.environ.get("SERVERLESS_MEMORY_MB", "2048"))
MAX_CONCURRENCY = int(os.environ.get("SERVERLESS_MAX_CONCURRENCY", "2"))


def handler(event, _context):
    detail = event["detail"]
    if detail.get("ModelApprovalStatus") != "Approved":
        return {"skipped": detail.get("ModelApprovalStatus")}

    package_arn = detail["ModelPackageArn"]
    version = package_arn.rsplit("/", 1)[-1]
    suffix = f"v{version}-{int(time.time())}"

    model_name = f"{ENDPOINT}-{suffix}"
    sm.create_model(ModelName=model_name, ExecutionRoleArn=ROLE_ARN,
                    PrimaryContainer={"ModelPackageName": package_arn})

    config_name = f"{ENDPOINT}-{suffix}"
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "ServerlessConfig": {"MemorySizeInMB": MEMORY_MB, "MaxConcurrency": MAX_CONCURRENCY},
        }],
        Tags=[{"Key": "model-package", "Value": package_arn}],
    )

    try:
        sm.describe_endpoint(EndpointName=ENDPOINT)
        sm.update_endpoint(EndpointName=ENDPOINT, EndpointConfigName=config_name)
        action = "updated"
    except ClientError as exc:
        if "Could not find endpoint" not in str(exc):
            raise
        sm.create_endpoint(EndpointName=ENDPOINT, EndpointConfigName=config_name)
        action = "created"

    print(f"{action} endpoint {ENDPOINT} -> {config_name} ({package_arn})")
    return {"endpoint": ENDPOINT, "config": config_name, "action": action}
