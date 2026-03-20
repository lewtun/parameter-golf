"""
Parameter-golf training job — runs on Hugging Face infrastructure.
Launched by launch_job.py via api.run_job() with the huggingface/trl Docker image
(torch + transformers prebuilt).
"""
import datetime, json, os, re, subprocess, sys, tempfile
from pathlib import Path
from huggingface_hub import batch_bucket_files, download_bucket_files

REPO_URL = "https://github.com/lewtun/parameter-golf"

bucket_id  = os.environ["BUCKET_ID"]
experiment = os.environ["EXPERIMENT_NAME"]


def update_config_status(status: str):
    tmp = Path(tempfile.mkdtemp())
    download_bucket_files(bucket_id, files=[(f"{experiment}/config.json", str(tmp / "c.json"))])
    config = json.loads((tmp / "c.json").read_text())
    config["status"] = status
    batch_bucket_files(
        bucket_id,
        add=[(json.dumps(config, indent=2).encode(), f"{experiment}/config.json")],
    )


def log(msg: str):
    print(f"[{datetime.datetime.utcnow().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log(f"train_job.py starting — experiment={experiment} bucket={bucket_id}")
    log(f"Python {sys.version}")

    workdir = Path("/workspace")
    workdir.mkdir(exist_ok=True)

    # Install extra deps not in the huggingface/trl base image
    # (torch, datasets, numpy, tqdm etc. are already prebuilt)
    log("Installing extra dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "tiktoken", "sentencepiece", "kernels", "trackio",
        "typing-extensions==4.15.0",
    ], check=True)
    log("Dependencies installed.")

    # Clone repo
    log("Cloning repo...")
    subprocess.run(["git", "clone", REPO_URL, str(workdir / "repo")], check=True)
    repo = workdir / "repo"
    log("Repo cloned.")

    # Download FineWeb data
    log("Downloading FineWeb data...")
    subprocess.run(
        [sys.executable, "data/cached_challenge_fineweb.py", "--train-shards", "10"],
        cwd=repo, check=True,
    )

    log("FineWeb data downloaded.")

    # Initialize Trackio for live metric tracking (imported after pip install)
    import trackio as wandb
    trackio_space = os.environ.get("TRACKIO_SPACE_ID")
    wandb.init(
        project="parameter-golf",
        name=experiment,
        config={
            "num_layers": os.environ.get("NUM_LAYERS"),
            "model_dim": os.environ.get("MODEL_DIM"),
            "matrix_lr": os.environ.get("MATRIX_LR"),
            "embed_lr": os.environ.get("EMBED_LR"),
            "bucket_id": bucket_id,
        },
        **({"space_id": trackio_space} if trackio_space else {}),
    )

    # Run training — env vars (NUM_LAYERS, MODEL_DIM, etc.) picked up by Hyperparameters class
    log("Starting torchrun (nproc_per_node=8)...")
    log_path = workdir / "train.log"
    log_lines = []

    proc = subprocess.Popen(
        ["torchrun", "--nproc_per_node=8", "train_gpt.py"],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
    )

    log_file = open(log_path, "w")
    for line in proc.stdout:
        print(line, end="", flush=True)  # stream to job logs live
        log_file.write(line)
        log_lines.append(line)

        # Parse and log training metrics to Trackio in real time
        step_m = re.search(
            r"step:(\d+)/\d+ train_loss:([\d.]+) train_time:(\d+)ms", line
        )
        if step_m:
            step = int(step_m.group(1))
            wandb.log({
                "train/loss": float(step_m.group(2)),
                "train/time_ms": int(step_m.group(3)),
            }, step=step)

        val_m = re.search(
            r"step:(\d+)/\d+ val_loss:([\d.]+) val_bpb:([\d.]+)", line
        )
        if val_m:
            step = int(val_m.group(1))
            wandb.log({
                "val/loss": float(val_m.group(2)),
                "val/bpb": float(val_m.group(3)),
            }, step=step)

    log_file.close()
    proc.wait()
    all_output = "".join(log_lines)

    # Parse final metrics from captured output
    val_bpb = val_loss = bytes_total = training_time_ms = None
    for line in all_output.splitlines():
        m = re.search(r"final_int8_zlib_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)", line)
        if m:
            val_loss, val_bpb = float(m.group(1)), float(m.group(2))
        m2 = re.search(r"Total submission size int8\+zlib: (\d+) bytes", line)
        if m2:
            bytes_total = int(m2.group(1))
        m3 = re.search(r"train_time:(\d+)ms", line)
        if m3:
            training_time_ms = int(m3.group(1))

    # Log final summary to Trackio
    summary = {"val_bpb": val_bpb, "val_loss": val_loss, "bytes_total": bytes_total,
               "training_time_ms": training_time_ms, "exit_code": proc.returncode}
    wandb.log({f"final/{k}": v for k, v in summary.items() if v is not None})
    wandb.finish()

    results = {
        "val_loss":          val_loss,
        "val_bpb":           val_bpb,
        "bytes_total":       bytes_total,
        "training_time_ms":  training_time_ms,
        "completed_at":      datetime.datetime.utcnow().isoformat() + "Z",
        "exit_code":         proc.returncode,
    }

    model_path = repo / "final_model.int8.ptz"
    files_to_upload = [
        (str(log_path),                           f"{experiment}/train.log"),
        (json.dumps(results, indent=2).encode(),  f"{experiment}/results.json"),
    ]
    if model_path.exists():
        files_to_upload.append((str(model_path), f"{experiment}/final_model.int8.ptz"))

    batch_bucket_files(bucket_id, add=files_to_upload)
    update_config_status("COMPLETED" if proc.returncode == 0 else "ERROR")
    print(f"Done. val_bpb={val_bpb}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
