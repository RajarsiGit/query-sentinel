"""Everything Query Sentinel needs in AWS, in one stack.

S3 bucket · SNS topic · SageMaker execution role · model package group ·
deployer Lambda + EventBridge rule (approve -> deploy) · scorer IAM policy.
The pipeline itself is upserted by `pipeline/deploy.py` so its definition can
change without a CloudFormation deploy.
"""
from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_events as events,
)
from aws_cdk import (
    aws_events_targets as targets,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_lambda as lambda_,
)
from aws_cdk import (
    aws_logs as logs,
)
from aws_cdk import (
    aws_s3 as s3,
)
from aws_cdk import (
    aws_sagemaker as sagemaker,
)
from aws_cdk import (
    aws_sns as sns,
)
from aws_cdk import (
    aws_sns_subscriptions as subs,
)
from constructs import Construct

PROJECT = "query-sentinel"


class SentinelStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, alert_email: str | None = None, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        bucket = s3.Bucket(
            self, "Bucket",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            versioned=True,
            lifecycle_rules=[
                s3.LifecycleRule(prefix="pipeline-runs/", expiration=Duration.days(90)),
                s3.LifecycleRule(prefix="scored-features/", expiration=Duration.days(30)),
                s3.LifecycleRule(noncurrent_version_expiration=Duration.days(7)),
            ],
            # Portfolio project: `cdk destroy` should leave nothing behind.
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        topic = sns.Topic(self, "Alerts", display_name="Query Sentinel alerts")
        if alert_email:
            topic.add_subscription(subs.EmailSubscription(alert_email))

        # ---- SageMaker execution role (pipeline, jobs, and the deployed model) ----
        sm_role = iam.Role(self, "SageMakerRole", assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"))
        bucket.grant_read_write(sm_role)
        sm_role.add_to_policy(iam.PolicyStatement(
            actions=["ecr:GetAuthorizationToken", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer",
                     "ecr:BatchCheckLayerAvailability",
                     "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams",
                     "cloudwatch:PutMetricData"],
            resources=["*"]))
        sm_role.add_to_policy(iam.PolicyStatement(
            actions=["sagemaker:CreateProcessingJob", "sagemaker:DescribeProcessingJob", "sagemaker:StopProcessingJob",
                     "sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob", "sagemaker:StopTrainingJob",
                     "sagemaker:CreateModelPackage", "sagemaker:DescribeModelPackage",
                     "sagemaker:CreateExperiment", "sagemaker:CreateTrial", "sagemaker:CreateTrialComponent",
                     "sagemaker:AssociateTrialComponent", "sagemaker:AddAssociation", "sagemaker:AddTags",
                     "sagemaker:DescribeExperiment", "sagemaker:DescribeTrial", "sagemaker:UpdateTrialComponent",
                     "sagemaker:ListTags"],
            resources=["*"]))
        sm_role.add_to_policy(iam.PolicyStatement(
            actions=["iam:PassRole"], resources=[sm_role.role_arn],
            conditions={"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}}))

        group = sagemaker.CfnModelPackageGroup(
            self, "ModelPackageGroup", model_package_group_name=PROJECT,
            model_package_group_description="Query Sentinel regression-onset classifiers")

        # ---- approve -> deploy ----
        deployer = lambda_.Function(
            self, "Deployer",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset("lambdas/deployer"),
            timeout=Duration.minutes(1),
            environment={"ENDPOINT_NAME": PROJECT, "MODEL_ROLE_ARN": sm_role.role_arn},
            log_retention=logs.RetentionDays.ONE_MONTH,
        )
        deployer.add_to_role_policy(iam.PolicyStatement(
            actions=["sagemaker:CreateModel", "sagemaker:CreateEndpointConfig", "sagemaker:CreateEndpoint",
                     "sagemaker:UpdateEndpoint", "sagemaker:DescribeEndpoint", "sagemaker:DescribeModelPackage",
                     "sagemaker:AddTags"],
            resources=["*"]))
        deployer.add_to_role_policy(iam.PolicyStatement(actions=["iam:PassRole"], resources=[sm_role.role_arn]))

        events.Rule(
            self, "OnModelApproved",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["SageMaker Model Package State Change"],
                detail={"ModelPackageGroupName": [PROJECT], "ModelApprovalStatus": ["Approved"]},
            ),
            targets=[targets.LambdaFunction(deployer, retry_attempts=2)],
        ).node.add_dependency(group)

        # ---- what the scorer needs (attach to an ECS task role, or your dev user) ----
        scorer_policy = iam.ManagedPolicy(
            self, "ScorerPolicy",
            statements=[
                iam.PolicyStatement(actions=["sagemaker:InvokeEndpoint", "sagemaker:DescribeEndpoint"],
                                    resources=[f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint/{PROJECT}"]),
                iam.PolicyStatement(actions=["sns:Publish"], resources=[topic.topic_arn]),
                iam.PolicyStatement(actions=["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
                                    resources=[bucket.bucket_arn, bucket.arn_for_objects("*")]),
            ],
        )

        for name, value in {
            "BucketName": bucket.bucket_name,
            "PipelineRoleArn": sm_role.role_arn,
            "AlertsTopicArn": topic.topic_arn,
            "EndpointName": PROJECT,
            "ScorerPolicyArn": scorer_policy.managed_policy_arn,
        }.items():
            CfnOutput(self, name, value=value)
