# HF Jobs + Buckets Example

This example shows how to use [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs) and [Buckets](https://huggingface.co/docs/hub/storage-buckets) to launch and track parameter-golf training experiments, plus a Gradio dashboard Space for monitoring.

## How it works

```
launch_job.py (local)
  ├── Creates bucket  hf://buckets/{username}/parameter-golf/
  ├── Writes          {experiment}/config.json  (status: LAUNCHING)
  ├── Submits         train_job.py via run_uv_job()
  └── Updates         {experiment}/config.json  (status: RUNNING, job_id, job_url)

train_job.py (cloud, via UV)
  ├── Installs torch + training deps
  ├── Clones the repo
  ├── Downloads FineWeb data
  ├── Runs torchrun train_gpt.py
  └── Uploads to bucket:
        {experiment}/train.log
        {experiment}/results.json
        {experiment}/final_model.int8.ptz   (if training succeeded)

dashboard/app.py (Gradio Space)
  └── Reads all experiments from bucket → status table + BPB chart
```

## Bucket layout

```
hf://buckets/{username}/parameter-golf/
└── {experiment_name}/
    ├── config.json          ← written by launcher (status: LAUNCHING → RUNNING → COMPLETED/ERROR)
    ├── train.log            ← uploaded by train_job.py after training
    ├── final_model.int8.ptz ← uploaded by train_job.py if training succeeded
    └── results.json         ← written by train_job.py on completion
```

## Setup

```bash
export HF_TOKEN=hf_...
pip install "huggingface_hub>=1.7.0"
```

## Usage

### Launch an experiment

```bash
# Baseline run on 8×A100
python launch_job.py --name baseline-9L --hardware a100x8

# 10-layer run on 8×H200
python launch_job.py --name my-10L-run --layers 10 --dim 512 --hardware h200x8

# Cheap test run (fewer GPUs — reduce --nproc_per_node accordingly)
python launch_job.py --name dev-test --layers 9 --hardware a10g-largex4
```

### Monitor

```bash
# Follow live logs
hf jobs logs <JOB_ID> -f

# List bucket contents
hf buckets ls {username}/parameter-golf -R
```

### Available hardware

| Flavor | GPUs | GPU RAM | Cost/hr |
|--------|------|---------|---------|
| `a100x8` | 8× A100 | 640 GB | $20.00 |
| `h200x8` | 8× H200 | 1128 GB | $40.00 |
| `l40sx8` | 8× L40S | 384 GB | $23.50 |
| `a100x4` | 4× A100 | 320 GB | $10.00 |
| `a10g-largex4` | 4× A10G | 96 GB | $5.00 |

Run `hf jobs hardware` to see the full list.

> **Note:** HF Jobs does not offer H100 instances. The competition requires reproducibility on 8×H100; `a100x8` is the closest available option. Both `a100x8` and `h200x8` have 8 GPUs so `--nproc_per_node=8` applies unchanged.

### Env vars passed to train_job.py

`train_job.py` passes these directly to `train_gpt.py`'s `Hyperparameters` class:

| CLI arg | Env var | Default |
|---------|---------|---------|
| `--layers` | `NUM_LAYERS` | 9 |
| `--dim` | `MODEL_DIM` | 512 |
| `--matrix-lr` | `MATRIX_LR` | 0.04 |
| `--embed-lr` | `EMBED_LR` | 0.6 |

## Dashboard

The `dashboard/` folder contains a Gradio app that reads all experiments from the bucket and shows a status table and BPB-over-time chart.

Set the `BUCKET_ID` environment variable (e.g. `username/parameter-golf`) before launching:

```bash
cd dashboard
BUCKET_ID=myuser/parameter-golf python app.py
```

To deploy as a Hugging Face Space, push `dashboard/` to a Space repo and set `BUCKET_ID` as a Space secret.
