"""Static checks on the pipeline JSON — catch broken references before SageMaker does."""
import json
import re

from pipeline.definition import build_definition

DEF = build_definition(bucket="b", role_arn="arn:aws:iam::123456789012:role/r", region="ap-south-1",
                       code_uri="s3://b/code/x")


def _all_steps(steps):
    for s in steps:
        yield s
        if s["Type"] == "Condition":
            yield from _all_steps(s["Arguments"]["IfSteps"])
            yield from _all_steps(s["Arguments"]["ElseSteps"])


def test_serialises():
    json.dumps(DEF)


def test_every_reference_resolves():
    text = json.dumps(DEF)
    steps = {s["Name"] for s in _all_steps(DEF["Steps"])}
    params = {p["Name"] for p in DEF["Parameters"]}
    for name in re.findall(r'"Get": "Steps\.([A-Za-z]+)\.', text):
        assert name in steps, f"unknown step {name}"
    for name in re.findall(r'"Get": "Parameters\.([A-Za-z]+)"', text):
        assert name in params, f"unknown parameter {name}"
    for dep in (d for s in _all_steps(DEF["Steps"]) for d in s.get("DependsOn", [])):
        assert dep in steps


def test_processing_output_references_exist():
    outputs = {s["Name"]: {o["OutputName"] for o in s["Arguments"]["ProcessingOutputConfig"]["Outputs"]}
               for s in _all_steps(DEF["Steps"]) if s["Type"] == "Processing"}
    for step, out in re.findall(r"Steps\.(\w+)\.ProcessingOutputConfig\.Outputs\['(\w+)'\]", json.dumps(DEF)):
        assert out in outputs[step], f"{step} has no output {out}"


def test_quality_gate_compares_against_heuristic():
    gate = next(s for s in DEF["Steps"] if s["Type"] == "Condition")
    paths = [c["RightValue"].get("Std:JsonGet", {}).get("Path") for c in gate["Arguments"]["Conditions"]
             if isinstance(c["RightValue"], dict)]
    assert "heuristic.pr_auc.value" in paths


def test_training_uses_spot_with_checkpoints():
    train = next(s for s in DEF["Steps"] if s["Type"] == "Training")["Arguments"]
    assert train["EnableManagedSpotTraining"] is True
    assert "CheckpointConfig" in train
    assert train["StoppingCondition"]["MaxWaitTimeInSeconds"] >= train["StoppingCondition"]["MaxRuntimeInSeconds"]


def test_hyperparameters_are_strings_or_params():
    hp = next(s for s in DEF["Steps"] if s["Type"] == "Training")["Arguments"]["HyperParameters"]
    for k, v in hp.items():
        assert isinstance(v, str) or "Get" in v, k


def test_new_models_require_manual_approval():
    reg = next(s for s in _all_steps(DEF["Steps"]) if s["Type"] == "RegisterModel")
    assert reg["Arguments"]["ModelApprovalStatus"] == "PendingManualApproval"
