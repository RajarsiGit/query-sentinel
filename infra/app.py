#!/usr/bin/env python3
import os

import aws_cdk as cdk

from stacks.sentinel_stack import SentinelStack

app = cdk.App()
SentinelStack(
    app, "QuerySentinel",
    alert_email=app.node.try_get_context("alertEmail"),
    env=cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
                        region=os.environ.get("CDK_DEFAULT_REGION", "ap-south-1")),
)
cdk.Tags.of(app).add("project", "query-sentinel")
app.synth()
