"""`sentinel` command line.

    sentinel seed                         # load the synthetic shop data
    sentinel generate --minutes 90        # workload + chaos + collector -> data/snapshots/run-*.jsonl
    sentinel build-dataset                # snapshots -> data/dataset.csv (features + labels)
    sentinel upload                       # dataset -> s3://$SENTINEL_BUCKET/datasets/<timestamp>/
    sentinel score [--loop 30]            # score live queries against the endpoint (or --model-file)
    sentinel drift --baseline s3://...    # PSI of logged live features vs training baseline
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import threading
import time
from pathlib import Path

from . import collector, dataset, workload
from .config import settings

log = logging.getLogger("sentinel")
DATA = Path("data")


def cmd_seed(a: argparse.Namespace) -> None:
    workload.seed(settings.dsn, customers=a.customers, orders=a.orders, events=a.events)


def cmd_generate(a: argparse.Namespace) -> None:
    """Run the workload, inject chaos, and collect snapshots, all for `--minutes`."""
    if a.reset:
        workload.seed(settings.dsn, customers=a.customers, orders=a.orders, events=a.events)
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = DATA / "snapshots" / f"run-{run_id}.jsonl"
    stop = threading.Event()
    threads, chaos_log = workload.run(settings.dsn, stop, workers=a.workers, rate_per_worker=a.rate,
                                      chaos_every_s=a.chaos_every, chaos_hold_s=a.chaos_hold)
    time.sleep(2)  # let workers connect before the first snapshot
    try:
        n = collector.collect(settings.dsn, out, a.interval, duration_s=int(a.minutes * 60), stop_event=stop)
    except KeyboardInterrupt:
        n = -1
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=60)
    (out.with_suffix(".chaos.json")).write_text(json.dumps(chaos_log, indent=2))
    log.info("wrote %s (%s snapshots, %d chaos events)", out, n, len(chaos_log))


def cmd_build_dataset(a: argparse.Namespace) -> None:
    files = sorted((DATA / "snapshots").glob("run-*.jsonl")) if not a.files else [Path(f) for f in a.files]
    if not files:
        sys.exit("no snapshot files found; run `sentinel generate` first")
    stats = dataset.build_dataset(files, Path(a.out), min_calls=settings.min_calls,
                                  baseline_windows=settings.baseline_windows,
                                  horizon=settings.horizon, ratio=settings.regression_ratio)
    log.info("dataset %s: %s", a.out, dataset.summarise(stats))


def cmd_upload(a: argparse.Namespace) -> None:
    import boto3
    if not settings.bucket:
        sys.exit("set SENTINEL_BUCKET (see `make outputs`)")
    key = f"datasets/{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}/dataset.csv"
    boto3.client("s3", region_name=settings.region).upload_file(a.file, settings.bucket, key)
    print(f"s3://{settings.bucket}/{key}")


def cmd_score(a: argparse.Namespace) -> None:
    from .scorer import EndpointPredictor, LocalPredictor, score_once
    predictor = (LocalPredictor(Path(a.model_file)) if a.model_file
                 else EndpointPredictor(settings.endpoint_name, settings.region))
    state = DATA / "scorer-state.jsonl"
    while True:
        results = score_once(settings, predictor, state, publish=not a.no_publish)
        for r in sorted(results, key=lambda r: -r["probability"])[: a.top]:
            print(f"{r['status']:<9} p={r['probability']:.2f}  {r['window_mean_ms']:8.2f}ms "
                  f"(base {r['baseline_ms']:.2f})  {r['query'][:90]}")
        if not a.loop:
            break
        print("-" * 80)
        time.sleep(a.loop)


def cmd_drift(a: argparse.Namespace) -> None:
    from . import drift
    if a.baseline.startswith("s3://"):
        import boto3
        bucket, key = a.baseline[5:].split("/", 1)
        baseline = json.loads(boto3.client("s3", region_name=settings.region)
                              .get_object(Bucket=bucket, Key=key)["Body"].read())
    else:
        baseline = json.loads(Path(a.baseline).read_text())
    if a.rows:
        rows = [json.loads(line) for line in Path(a.rows).read_text().splitlines() if line.strip()]
    else:
        rows = drift.load_rows_from_s3(settings.bucket, "scored-features/", settings.region)
    report = drift.drift_report(baseline, rows)
    for name, value in report.items():
        flag = "DRIFT" if value > drift.PSI_ALERT else ""
        print(f"{name:<26} PSI={value:6.3f} {flag}")
    drifted = [k for k, v in report.items() if v > drift.PSI_ALERT]
    if drifted and settings.sns_topic_arn and not a.no_publish:
        import boto3
        boto3.client("sns", region_name=settings.region).publish(
            TopicArn=settings.sns_topic_arn, Subject="Query Sentinel: feature drift detected",
            Message="PSI > 0.2 for: " + ", ".join(drifted) + "\nConsider re-running the training pipeline.")
    sys.exit(1 if drifted else 0)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="sentinel")
    sub = p.add_subparsers(dest="cmd", required=True)

    def size_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--customers", type=int, default=20_000)
        sp.add_argument("--orders", type=int, default=100_000)
        sp.add_argument("--events", type=int, default=200_000)

    sp = sub.add_parser("seed", help="load synthetic data")
    size_args(sp)
    sp.set_defaults(func=cmd_seed)

    sp = sub.add_parser("generate", help="workload + chaos + snapshot collection")
    size_args(sp)
    sp.add_argument("--minutes", type=float, default=90)
    sp.add_argument("--interval", type=int, default=settings.snapshot_interval_s)
    sp.add_argument("--workers", type=int, default=8)
    sp.add_argument("--rate", type=float, default=40.0, help="queries/sec per worker")
    sp.add_argument("--chaos-every", type=int, default=300)
    sp.add_argument("--chaos-hold", type=int, default=150)
    sp.add_argument("--reset", action="store_true", help="re-seed before the run")
    sp.set_defaults(func=cmd_generate)

    sp = sub.add_parser("build-dataset", help="snapshots -> labelled CSV")
    sp.add_argument("--out", default=str(DATA / "dataset.csv"))
    sp.add_argument("files", nargs="*")
    sp.set_defaults(func=cmd_build_dataset)

    sp = sub.add_parser("upload", help="upload dataset CSV to S3")
    sp.add_argument("--file", default=str(DATA / "dataset.csv"))
    sp.set_defaults(func=cmd_upload)

    sp = sub.add_parser("score", help="score live queries")
    sp.add_argument("--model-file", help="use a local xgboost-model file instead of the endpoint")
    sp.add_argument("--loop", type=int, default=0, help="repeat every N seconds")
    sp.add_argument("--top", type=int, default=10)
    sp.add_argument("--no-publish", action="store_true", help="skip SNS and S3 logging")
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("drift", help="PSI drift report")
    sp.add_argument("--baseline", required=True, help="path or s3:// URI of baseline_stats.json")
    sp.add_argument("--rows", help="local JSON-lines feature log instead of S3")
    sp.add_argument("--no-publish", action="store_true")
    sp.set_defaults(func=cmd_drift)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
