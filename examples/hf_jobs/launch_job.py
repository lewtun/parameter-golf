#!/usr/bin/env python3
"""Launch a parameter-golf training experiment using HF Jobs.

Usage:
    hf auth login
    python launch_job.py --name "my-10L-run" --layers 10 --dim 512
"""
import argparse, datetime, json, os

os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
from pathlib import Path
from huggingface_hub import HfApi, create_bucket, batch_bucket_files, run_uv_job, whoami
from huggingface_hub.utils import get_token

BUCKET_REPO = "parameter-golf"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True, help="Unique experiment name (bucket prefix)")
    p.add_argument("--namespace", default=None, help="HF username or org (defaults to your account)")
    p.add_argument("--layers",    type=int,   default=9)
    p.add_argument("--dim",       type=int,   default=512)
    p.add_argument("--matrix-lr", type=float, default=0.04)
    p.add_argument("--embed-lr",  type=float, default=0.6)
    p.add_argument("--hardware",  default="a100x8",
                   help="HF flavor (run `hf jobs hardware` to list options; "
                        "use a100x8 for 8×A100 or h200x8 for 8×H200)")
    p.add_argument("--image", default=None,
                   help="Custom Docker image (e.g. huggingface/trl)")
    p.add_argument("--timeout",   default="20m")
    p.add_argument("-d", "--detach", action="store_true",
                   help="Print job ID and exit without streaming logs")
    p.add_argument("--trackio-space", default="parameter-golf-experiments",
                   help="Trackio Space name for live metrics (default: parameter-golf-experiments)")
    p.add_argument("--no-trackio", action="store_true",
                   help="Disable Trackio logging")
    return p.parse_args()


def main():
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN") or get_token()
    if not hf_token:
        raise SystemExit("Set HF_TOKEN or run `hf auth login` first.")

    namespace = args.namespace or whoami()["name"]
    bucket_id = f"{namespace}/{BUCKET_REPO}"

    create_bucket(bucket_id, exist_ok=True)

    config = {
        "name":        args.name,
        "hardware":    args.hardware,
        "num_layers":  args.layers,
        "model_dim":   args.dim,
        "matrix_lr":   args.matrix_lr,
        "embed_lr":    args.embed_lr,
        "launched_at": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
        "status":      "LAUNCHING",
    }
    batch_bucket_files(
        bucket_id,
        add=[(json.dumps(config, indent=2).encode(), f"{args.name}/config.json")],
    )

    env = {
        "PYTHONUNBUFFERED": "1",
        "EXPERIMENT_NAME": args.name,
        "BUCKET_ID":       bucket_id,
        "NUM_LAYERS":      str(args.layers),
        "MODEL_DIM":       str(args.dim),
        "MATRIX_LR":       str(args.matrix_lr),
        "EMBED_LR":        str(args.embed_lr),
    }
    if args.image:
        env["DOCKER_IMAGE"] = args.image
    if not args.no_trackio and args.trackio_space:
        env["TRACKIO_SPACE_ID"] = f"{namespace}/{args.trackio_space}"

    job_kwargs = dict(
        script=Path(__file__).parent / "train_job.py",
        flavor=args.hardware,
        env=env,
        secrets={"HF_TOKEN": hf_token},
        timeout=args.timeout,
        namespace=namespace,
    )
    if args.image:
        job_kwargs["image"] = args.image

    job = run_uv_job(**job_kwargs)

    config.update({"job_id": job.id, "job_url": job.url, "status": "RUNNING"})
    batch_bucket_files(
        bucket_id,
        add=[(json.dumps(config, indent=2).encode(), f"{args.name}/config.json")],
    )

    print(f"Launched:  {job.url}")
    print(f"Bucket:    hf://buckets/{bucket_id}/{args.name}/")

    if args.detach:
        print(f"Follow:    hf jobs logs {job.id} -f")
        return

    # Stream logs until job completes
    print(f"\n--- Streaming logs for {job.id} (Ctrl+C to detach) ---\n")
    api = HfApi(token=hf_token)
    try:
        for log in api.fetch_job_logs(job_id=job.id, follow=True):
            print(log)
    except KeyboardInterrupt:
        print(f"\nDetached. Follow:  hf jobs logs {job.id} -f")


if __name__ == "__main__":
    main()
