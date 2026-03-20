# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface-hub>=1.7.0",
#   "numpy",
#   "tqdm",
# ]
# ///
"""
Parameter-golf training job — runs on Hugging Face infrastructure.
Launched by launch_job.py via run_uv_job().
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


def main():
    workdir = Path("/workspace")
    workdir.mkdir(exist_ok=True)

    # Install training deps (torch, etc.) — use uv since the UV runtime doesn't ship pip
    subprocess.run([
        "uv", "pip", "install", "--quiet",
        "torch", "datasets", "tiktoken", "sentencepiece",
        "kernels", "typing-extensions==4.15.0",
    ], check=True)

    # Clone repo
    subprocess.run(["git", "clone", REPO_URL, str(workdir / "repo")], check=True)
    repo = workdir / "repo"

    # Download FineWeb data
    subprocess.run(
        [sys.executable, "data/cached_challenge_fineweb.py", "--train-shards", "10"],
        cwd=repo, check=True,
    )

    # Run training — env vars (NUM_LAYERS, MODEL_DIM, etc.) picked up by Hyperparameters class
    log_path = workdir / "train.log"
    result = subprocess.run(
        ["torchrun", "--nproc_per_node=8", "train_gpt.py"],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(result.stdout)
    print(result.stdout)  # stream to job logs

    # Parse final metrics
    val_bpb = val_loss = bytes_total = training_time_ms = None
    for line in result.stdout.splitlines():
        m = re.search(r"final_int8_zlib_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)", line)
        if m:
            val_loss, val_bpb = float(m.group(1)), float(m.group(2))
        m2 = re.search(r"Total submission size int8\+zlib: (\d+) bytes", line)
        if m2:
            bytes_total = int(m2.group(1))
        m3 = re.search(r"train_time:(\d+)ms", line)
        if m3:
            training_time_ms = int(m3.group(1))

    results = {
        "val_loss":          val_loss,
        "val_bpb":           val_bpb,
        "bytes_total":       bytes_total,
        "training_time_ms":  training_time_ms,
        "completed_at":      datetime.datetime.utcnow().isoformat() + "Z",
        "exit_code":         result.returncode,
    }

    model_path = repo / "final_model.int8.ptz"
    files_to_upload = [
        (str(log_path),                           f"{experiment}/train.log"),
        (json.dumps(results, indent=2).encode(),  f"{experiment}/results.json"),
    ]
    if model_path.exists():
        files_to_upload.append((str(model_path), f"{experiment}/final_model.int8.ptz"))

    batch_bucket_files(bucket_id, add=files_to_upload)
    update_config_status("COMPLETED" if result.returncode == 0 else "ERROR")
    print(f"Done. val_bpb={val_bpb}")


if __name__ == "__main__":
    main()
