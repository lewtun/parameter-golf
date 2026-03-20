import json, os, tempfile
from pathlib import Path
import gradio as gr
import plotly.graph_objects as go
from huggingface_hub import list_bucket_tree, download_bucket_files

BUCKET_ID = os.environ.get("BUCKET_ID", "username/parameter-golf")


def load_experiments():
    exps = []
    for item in list_bucket_tree(BUCKET_ID, recursive=False):
        if item.type != "directory":
            continue
        prefix = item.path.rstrip("/")
        tmp = Path(tempfile.mkdtemp())
        try:
            download_bucket_files(BUCKET_ID, files=[(f"{prefix}/config.json", str(tmp / "c.json"))])
            config = json.loads((tmp / "c.json").read_text())
        except Exception:
            continue
        results = None
        try:
            download_bucket_files(BUCKET_ID, files=[(f"{prefix}/results.json", str(tmp / "r.json"))])
            results = json.loads((tmp / "r.json").read_text())
        except Exception:
            pass
        exps.append({
            "name":        config.get("name", prefix),
            "status":      config.get("status", "UNKNOWN"),
            "hardware":    config.get("hardware", ""),
            "num_layers":  config.get("num_layers", ""),
            "model_dim":   config.get("model_dim", ""),
            "launched_at": config.get("launched_at", ""),
            "job_url":     config.get("job_url", ""),
            "val_bpb":     results["val_bpb"]     if results else None,
            "bytes_total": results["bytes_total"] if results else None,
        })
    return exps


def refresh():
    exps = load_experiments()
    rows = [
        [e["name"], e["status"], e["hardware"], e["num_layers"], e["model_dim"],
         f"{e['val_bpb']:.4f}" if e["val_bpb"] else "-",
         f"{e['bytes_total']:,}" if e["bytes_total"] else "-",
         e["launched_at"][:19] if e["launched_at"] else "-",
         e["job_url"] or "-"]
        for e in sorted(exps, key=lambda x: x["val_bpb"] or 9999)
    ]
    completed = sorted([e for e in exps if e["val_bpb"]], key=lambda x: x["launched_at"])
    fig = go.Figure(go.Scatter(
        x=[e["launched_at"][:19] for e in completed],
        y=[e["val_bpb"] for e in completed],
        mode="markers+text",
        text=[e["name"] for e in completed],
        textposition="top center",
        marker=dict(size=10),
    ))
    fig.update_layout(title="val_bpb over time (lower = better)", yaxis=dict(autorange="reversed"))
    return rows, fig


HEADERS = ["Name", "Status", "Hardware", "Layers", "Dim", "val_bpb", "Bytes", "Launched", "Job URL"]

with gr.Blocks(title="Parameter Golf Dashboard") as demo:
    gr.Markdown(f"# Parameter Golf - Experiment Dashboard\n`hf://buckets/{BUCKET_ID}`")
    df   = gr.Dataframe(headers=HEADERS, label="Experiments")
    plot = gr.Plot(label="BPB Progress")
    gr.Button("Refresh").click(fn=refresh, outputs=[df, plot])
    gr.Timer(value=30).tick(fn=refresh, outputs=[df, plot])
    demo.load(fn=refresh, outputs=[df, plot])

demo.launch()
