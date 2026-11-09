"""Upsert and run the SageMaker pipeline.

    python -m pipeline.deploy upsert                      # upload step code, create/update pipeline
    python -m pipeline.deploy start --data s3://.../       # start an execution
    python -m pipeline.deploy status                      # latest executions and step states
    python -m pipeline.deploy approve                     # approve the newest pending model package

Reads SENTINEL_BUCKET, SENTINEL_PIPELINE_ROLE_ARN, AWS_REGION (exported by `make outputs`).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import boto3

from .definition import MODEL_PACKAGE_GROUP, PIPELINE_NAME, build_definition

SCRIPTS = Path(__file__).parent / "scripts"


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"{name} is not set (run `make outputs` and export the values)")
    return value


def upload_code(s3, bucket: str) -> str:
    """Content-addressed upload: a code change produces a new prefix, so old runs stay reproducible."""
    files = sorted(SCRIPTS.glob("*.py"))
    digest = hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:12]
    prefix = f"code/{digest}"
    for f in files:
        s3.upload_file(str(f), bucket, f"{prefix}/{f.name}")
    return f"s3://{bucket}/{prefix}"


def cmd_upsert(_: argparse.Namespace) -> None:
    region, bucket, role = _env("AWS_REGION"), _env("SENTINEL_BUCKET"), _env("SENTINEL_PIPELINE_ROLE_ARN")
    code_uri = upload_code(boto3.client("s3", region_name=region), bucket)
    definition = json.dumps(build_definition(bucket=bucket, role_arn=role, region=region, code_uri=code_uri))
    sm = boto3.client("sagemaker", region_name=region)
    kwargs = {"PipelineName": PIPELINE_NAME, "PipelineDefinition": definition, "RoleArn": role,
              "PipelineDescription": "Query Sentinel: predict PostgreSQL query regressions"}
    try:
        sm.describe_pipeline(PipelineName=PIPELINE_NAME)
        sm.update_pipeline(**kwargs)
        print(f"updated pipeline {PIPELINE_NAME} (code {code_uri})")
    except sm.exceptions.ResourceNotFound:
        sm.create_pipeline(**kwargs)
        print(f"created pipeline {PIPELINE_NAME} (code {code_uri})")


def _scale_pos_weight(local_csv: Path) -> str | None:
    if not local_csv.exists():
        return None
    with local_csv.open() as fh:
        labels = [int(r["label"]) for r in csv.DictReader(fh)]
    pos = sum(labels)
    return f"{(len(labels) - pos) / max(pos, 1):.3f}" if labels else None


def cmd_start(a: argparse.Namespace) -> None:
    sm = boto3.client("sagemaker", region_name=_env("AWS_REGION"))
    params = [{"Name": "InputDataUri", "Value": a.data}]
    spw = a.scale_pos_weight or _scale_pos_weight(Path(a.local_dataset))
    if spw:
        params.append({"Name": "ScalePosWeight", "Value": spw})
    for kv in a.param or []:
        k, v = kv.split("=", 1)
        params.append({"Name": k, "Value": v})
    resp = sm.start_pipeline_execution(PipelineName=PIPELINE_NAME, PipelineParameters=params)
    print(resp["PipelineExecutionArn"])


def cmd_status(_: argparse.Namespace) -> None:
    sm = boto3.client("sagemaker", region_name=_env("AWS_REGION"))
    for ex in sm.list_pipeline_executions(PipelineName=PIPELINE_NAME, MaxResults=3)["PipelineExecutionSummaries"]:
        print(f"{ex['StartTime']:%Y-%m-%d %H:%M}  {ex['PipelineExecutionStatus']:<10} {ex['PipelineExecutionArn']}")
        arn = ex["PipelineExecutionArn"]
        steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=arn)["PipelineExecutionSteps"]
        for s in reversed(steps):
            reason = f"  ({s['FailureReason'][:120]})" if s.get("FailureReason") else ""
            print(f"    {s['StepName']:<12} {s['StepStatus']}{reason}")


def cmd_approve(_: argparse.Namespace) -> None:
    sm = boto3.client("sagemaker", region_name=_env("AWS_REGION"))
    pkgs = sm.list_model_packages(ModelPackageGroupName=MODEL_PACKAGE_GROUP,
                                  ModelApprovalStatus="PendingManualApproval",
                                  SortBy="CreationTime", SortOrder="Descending",
                                  MaxResults=1)["ModelPackageSummaryList"]
    if not pkgs:
        sys.exit("no pending model packages")
    arn = pkgs[0]["ModelPackageArn"]
    sm.update_model_package(ModelPackageArn=arn, ModelApprovalStatus="Approved",
                            ApprovalDescription="approved via pipeline/deploy.py")
    print(f"approved {arn} — the deployer Lambda will roll it out to the serverless endpoint")


def main() -> None:
    p = argparse.ArgumentParser(prog="pipeline.deploy")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("upsert").set_defaults(func=cmd_upsert)
    sp = sub.add_parser("start")
    sp.add_argument("--data", required=True, help="s3:// prefix of the dataset CSV(s)")
    sp.add_argument("--scale-pos-weight")
    sp.add_argument("--local-dataset", default="data/dataset.csv",
                    help="used to compute ScalePosWeight when not given")
    sp.add_argument("--param", action="append", help="extra Name=Value pipeline parameters")
    sp.set_defaults(func=cmd_start)
    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("approve").set_defaults(func=cmd_approve)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
