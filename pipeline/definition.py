"""SageMaker Pipeline definition, written as the raw pipeline-definition JSON.

Why raw JSON instead of the Python SDK: the SDK went from v2 to v3 with breaking
changes (Estimator -> ModelTrainer, etc.), while this schema did not. Each step's
`Arguments` is exactly the request body of the matching Create*Job API, so the
definition is easy to review and independent of SDK churn.

    Preprocess (Processing) -> Train (Training, Spot) -> Evaluate (Processing)
      -> QualityGate (Condition) -> Register (RegisterModel) | Rejected (Fail)
"""
from __future__ import annotations

import json
import os
from typing import Any

PIPELINE_NAME = "query-sentinel"
MODEL_PACKAGE_GROUP = "query-sentinel"

XGBOOST_VERSION = os.environ.get("XGBOOST_VERSION", "1.7-1")
SKLEARN_VERSION = os.environ.get("SKLEARN_VERSION", "1.2-1")

# Fallback registry accounts for the first-party sklearn/xgboost images. The SDK lookup
# (below) is preferred; override with XGBOOST_IMAGE / SKLEARN_IMAGE if your region is missing.
_IMAGE_ACCOUNTS = {
    "us-east-1": "683313688378", "us-east-2": "257758044811", "us-west-2": "246618743249",
    "ca-central-1": "341280168497", "eu-west-1": "141502667606", "ap-south-1": "720646828776",
    "ap-southeast-1": "121021644041", "ap-southeast-2": "783357654285",
}


def image_uri(framework: str, region: str) -> str:
    override = os.environ.get(f"{framework.upper()}_IMAGE")
    if override:
        return override
    version = XGBOOST_VERSION if framework == "xgboost" else SKLEARN_VERSION
    for mod in ("sagemaker.core.image_uris", "sagemaker.image_uris"):  # SDK v3, then v2
        try:
            retrieve = __import__(mod, fromlist=["retrieve"]).retrieve
            return retrieve(framework=framework, region=region, version=version,
                            **({"py_version": "py3", "instance_type": "ml.m5.large"} if framework == "sklearn" else {}))
        except Exception:
            continue
    if region not in _IMAGE_ACCOUNTS:
        raise SystemExit(f"no image account for {region}; set {framework.upper()}_IMAGE")
    repo = ("sagemaker-xgboost" if framework == "xgboost" else "sagemaker-scikit-learn")
    tag = version if framework == "xgboost" else f"{version}-cpu-py3"
    return f"{_IMAGE_ACCOUNTS[region]}.dkr.ecr.{region}.amazonaws.com/{repo}:{tag}"


# ---- tiny expression helpers -------------------------------------------------
def param(name: str) -> dict:
    return {"Get": f"Parameters.{name}"}


def get(path: str) -> dict:
    return {"Get": path}


def join(*values: Any, on: str = "/") -> dict:
    return {"Std:Join": {"On": on, "Values": list(values)}}


def run_path(bucket: str, *parts: str) -> dict:
    """s3://bucket/pipeline-runs/<execution-id>/<parts...> — unique per execution."""
    return join(f"s3://{bucket}/pipeline-runs", get("Execution.PipelineExecutionId"), *parts)


def processing_output(name: str, local: str, bucket: str) -> dict:
    return {"OutputName": name, "AppManaged": False,
            "S3Output": {"S3Uri": run_path(bucket, name), "LocalPath": local, "S3UploadMode": "EndOfJob"}}


def processing_input(name: str, s3_uri: Any, local: str) -> dict:
    return {"InputName": name, "AppManaged": False,
            "S3Input": {"S3Uri": s3_uri, "LocalPath": local, "S3DataType": "S3Prefix",
                        "S3InputMode": "File", "S3DataDistributionType": "FullyReplicated"}}


def output_uri(step: str, output: str) -> dict:
    return get(f"Steps.{step}.ProcessingOutputConfig.Outputs['{output}'].S3Output.S3Uri")


def eval_metric(path: str) -> dict:
    return {"Std:JsonGet": {"PropertyFile": get("Steps.Evaluate.PropertyFiles.EvaluationReport"), "Path": path}}


# ---- definition ----------------------------------------------------------------
def build_definition(*, bucket: str, role_arn: str, region: str, code_uri: str) -> dict:
    sklearn_img = image_uri("sklearn", region)
    xgb_img = image_uri("xgboost", region)
    proc_resources = {"ClusterConfig": {"InstanceType": "ml.m5.large", "InstanceCount": 1, "VolumeSizeInGB": 10}}
    code_dir = "/opt/ml/processing/input/code"

    preprocess = {
        "Name": "Preprocess", "Type": "Processing",
        "Arguments": {
            "ProcessingResources": proc_resources,
            "AppSpecification": {"ImageUri": sklearn_img,
                                 "ContainerEntrypoint": ["python3", f"{code_dir}/preprocess.py"]},
            "RoleArn": role_arn,
            "ProcessingInputs": [
                processing_input("data", param("InputDataUri"), "/opt/ml/processing/input/data"),
                processing_input("code", f"{code_uri}/preprocess.py", code_dir),
            ],
            "ProcessingOutputConfig": {"Outputs": [
                processing_output("train", "/opt/ml/processing/train", bucket),
                processing_output("validation", "/opt/ml/processing/validation", bucket),
                processing_output("test", "/opt/ml/processing/test", bucket),
                processing_output("baseline", "/opt/ml/processing/baseline", bucket),
            ]},
            "StoppingCondition": {"MaxRuntimeInSeconds": 1800},
        },
    }

    def channel(name: str) -> dict:
        return {"ChannelName": name, "ContentType": "text/csv",
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3DataDistributionType": "FullyReplicated",
                                                "S3Uri": output_uri("Preprocess", name)}}}

    train = {
        "Name": "Train", "Type": "Training",
        "Arguments": {
            "AlgorithmSpecification": {"TrainingImage": xgb_img, "TrainingInputMode": "File"},
            "RoleArn": role_arn,
            "InputDataConfig": [channel("train"), channel("validation")],
            "OutputDataConfig": {"S3OutputPath": run_path(bucket, "model")},
            "ResourceConfig": {"InstanceType": param("TrainInstanceType"), "InstanceCount": 1, "VolumeSizeInGB": 10},
            "HyperParameters": {
                "objective": "binary:logistic", "eval_metric": "aucpr",
                "num_round": param("NumRound"), "max_depth": param("MaxDepth"), "eta": param("Eta"),
                "subsample": "0.8", "colsample_bytree": "0.8", "min_child_weight": "3",
                "scale_pos_weight": param("ScalePosWeight"), "early_stopping_rounds": "20",
            },
            # Managed Spot: ~60-70% cheaper; checkpoints let an interrupted job resume.
            "EnableManagedSpotTraining": True,
            "CheckpointConfig": {"S3Uri": run_path(bucket, "checkpoints")},
            "StoppingCondition": {"MaxRuntimeInSeconds": 1800, "MaxWaitTimeInSeconds": 3600},
        },
        "DependsOn": ["Preprocess"],
    }

    evaluate = {
        "Name": "Evaluate", "Type": "Processing",
        "Arguments": {
            "ProcessingResources": proc_resources,
            "AppSpecification": {"ImageUri": xgb_img,
                                 "ContainerEntrypoint": ["python3", f"{code_dir}/evaluate.py"],
                                 "ContainerArguments": ["--threshold", param("AlertThreshold")]},
            "RoleArn": role_arn,
            "ProcessingInputs": [
                processing_input("model", get("Steps.Train.ModelArtifacts.S3ModelArtifacts"),
                                 "/opt/ml/processing/model"),
                processing_input("test", output_uri("Preprocess", "test"), "/opt/ml/processing/test"),
                processing_input("code", f"{code_uri}/evaluate.py", code_dir),
            ],
            "ProcessingOutputConfig": {"Outputs": [
                processing_output("evaluation", "/opt/ml/processing/evaluation", bucket),
            ]},
            "StoppingCondition": {"MaxRuntimeInSeconds": 1800},
        },
        "PropertyFiles": [{"PropertyFileName": "EvaluationReport", "OutputName": "evaluation",
                           "FilePath": "evaluation.json"}],
        "DependsOn": ["Train"],
    }

    register = {
        "Name": "Register", "Type": "RegisterModel",
        "Arguments": {
            "ModelPackageGroupName": MODEL_PACKAGE_GROUP,
            "ModelPackageDescription": "Query Sentinel regression-onset classifier (XGBoost)",
            "ModelApprovalStatus": "PendingManualApproval",
            "InferenceSpecification": {
                "Containers": [{"Image": xgb_img, "ModelDataUrl": get("Steps.Train.ModelArtifacts.S3ModelArtifacts")}],
                "SupportedContentTypes": ["text/csv"], "SupportedResponseMIMETypes": ["text/csv"],
            },
            "ModelMetrics": {"ModelQuality": {"Statistics": {
                "ContentType": "application/json",
                "S3Uri": join(output_uri("Evaluate", "evaluation"), "evaluation.json")}}},
            # The drift job finds its baseline from the approved package.
            "CustomerMetadataProperties": {
                "baseline_stats_uri": join(output_uri("Preprocess", "baseline"), "baseline_stats.json"),
                "pipeline_execution": get("Execution.PipelineExecutionArn"),
            },
        },
    }

    rejected = {
        "Name": "Rejected", "Type": "Fail",
        "Arguments": {"ErrorMessage": join("Model below quality gate. PR-AUC:", eval_metric("metrics.pr_auc.value"),
                                           "heuristic PR-AUC:", eval_metric("heuristic.pr_auc.value"), on=" ")},
    }

    gate = {
        "Name": "QualityGate", "Type": "Condition",
        "Arguments": {
            "Conditions": [  # all must hold
                {"Type": "GreaterThanOrEqualTo", "LeftValue": eval_metric("metrics.pr_auc.value"),
                 "RightValue": param("MinPrAuc")},
                {"Type": "GreaterThanOrEqualTo", "LeftValue": eval_metric("metrics.pr_auc.value"),
                 "RightValue": eval_metric("heuristic.pr_auc.value")},
            ],
            "IfSteps": [register],
            "ElseSteps": [rejected],
        },
        "DependsOn": ["Evaluate"],
    }

    return {
        "Version": "2020-12-01",
        "Metadata": {},
        "Parameters": [
            {"Name": "InputDataUri", "Type": "String", "DefaultValue": f"s3://{bucket}/datasets/latest/"},
            {"Name": "TrainInstanceType", "Type": "String", "DefaultValue": "ml.m5.xlarge"},
            {"Name": "NumRound", "Type": "String", "DefaultValue": "300"},
            {"Name": "MaxDepth", "Type": "String", "DefaultValue": "5"},
            {"Name": "Eta", "Type": "String", "DefaultValue": "0.1"},
            {"Name": "ScalePosWeight", "Type": "String", "DefaultValue": "5"},
            {"Name": "AlertThreshold", "Type": "String", "DefaultValue": "0.7"},
            {"Name": "MinPrAuc", "Type": "Float", "DefaultValue": 0.4},
        ],
        "PipelineExperimentConfig": {"ExperimentName": get("Execution.PipelineName"),
                                     "TrialName": get("Execution.PipelineExecutionId")},
        "Steps": [preprocess, train, evaluate, gate],
    }


if __name__ == "__main__":
    print(json.dumps(build_definition(bucket="example-bucket", role_arn="arn:aws:iam::123456789012:role/example",
                                      region="ap-south-1", code_uri="s3://example-bucket/code/abc"), indent=2))
