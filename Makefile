SHELL := /bin/bash
STACK := QuerySentinel
MINUTES ?= 90

.PHONY: help up down install seed data dataset test lint infra outputs pipeline train status approve score score-local drift destroy

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

# ---- local data plane (free) ----
up:              ## start Postgres 16 + Grafana in Docker
	docker compose up -d --wait
down:            ## stop local services
	docker compose down
install:         ## install the package with dev extras
	pip install -e ".[dev]"
seed:            ## load synthetic shop data
	sentinel seed
data:            ## run workload + chaos + collector for $(MINUTES) minutes
	sentinel generate --minutes $(MINUTES) --reset
dataset:         ## snapshots -> labelled data/dataset.csv
	sentinel build-dataset

# ---- quality ----
test:            ## unit tests (no AWS needed)
	pytest -q
lint:
	ruff check .

# ---- AWS ----
infra:           ## deploy the CDK stack (bucket, roles, registry, deployer)
	cd infra && pip install -q -r requirements.txt && cdk deploy --require-approval never $(if $(EMAIL),-c alertEmail=$(EMAIL))
outputs:         ## print env exports from the stack outputs; use: eval "$$(make -s outputs)"
	@aws cloudformation describe-stacks --stack-name $(STACK) --query 'Stacks[0].Outputs' --output json | \
	python3 -c 'import json,sys; m={"BucketName":"SENTINEL_BUCKET","PipelineRoleArn":"SENTINEL_PIPELINE_ROLE_ARN","AlertsTopicArn":"SENTINEL_TOPIC_ARN","EndpointName":"SENTINEL_ENDPOINT"}; [print(f"export {m[o[\"OutputKey\"]]}={o[\"OutputValue\"]}") for o in json.load(sys.stdin) if o["OutputKey"] in m]'
pipeline:        ## upload step code and create/update the SageMaker pipeline
	python -m pipeline.deploy upsert
train: dataset   ## upload dataset and start a pipeline execution
	python -m pipeline.deploy start --data $$(sentinel upload)
status:          ## latest pipeline executions
	python -m pipeline.deploy status
approve:         ## approve newest pending model -> deployer Lambda rolls it out
	python -m pipeline.deploy approve
score:           ## score live queries via the endpoint every 30s
	sentinel score --loop 30
score-local:     ## score with a local model file: make score-local MODEL=path/to/xgboost-model
	sentinel score --loop 30 --no-publish --model-file $(MODEL)
drift:           ## PSI drift: make drift BASELINE=s3://.../baseline_stats.json
	sentinel drift --baseline $(BASELINE)
destroy:         ## delete endpoint + stack (everything)
	-aws sagemaker delete-endpoint --endpoint-name query-sentinel
	-aws sagemaker delete-pipeline --pipeline-name query-sentinel
	-for arn in $$(aws sagemaker list-model-packages --model-package-group-name query-sentinel \
	    --query 'ModelPackageSummaryList[].ModelPackageArn' --output text); do \
	    aws sagemaker delete-model-package --model-package-name $$arn; done
	cd infra && cdk destroy --force
