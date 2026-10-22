"""EcoSort live-camera demo server.

    python app/server.py            # then open http://localhost:8000
"""
import glob
import io
import json
import os
import sys
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from models import load_checkpoint  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {}

for path in sorted(glob.glob(os.path.join(ROOT, "models", "*.pt"))):
    model, ckpt = load_checkpoint(path, DEVICE)
    size = ckpt["img_size"]
    metrics_path = os.path.join(ROOT, "results", ckpt["arch"], "metrics.json")
    test = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        test = {"test_acc": m["test"]["accuracy"], "test_f1": m["test"]["macro_f1"]}
    cal = ckpt.get("calibration", {"temperature": 1.0, "bias": [0.0] * len(ckpt["classes"])})
    MODELS[ckpt["arch"]] = {
        "model": model,
        "classes": ckpt["classes"],
        "temperature": cal["temperature"],
        "bias": torch.tensor(cal["bias"], device=DEVICE),
        "tf": T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size), T.ToTensor(), T.Normalize(ckpt["mean"], ckpt["std"])]),
        "info": {"name": ckpt["arch"], "val_acc": ckpt["val_acc"], "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
                 "calibrated": "calibration" in ckpt, **test},
    }
    print(f"loaded {ckpt['arch']} on {DEVICE}")

if not MODELS:
    sys.exit("No models found in models/. Train one first: python src/train.py")

# best model (by test accuracy, else val accuracy) is the default
DEFAULT = max(MODELS, key=lambda k: MODELS[k]["info"].get("test_acc", MODELS[k]["info"]["val_acc"]))

app = FastAPI(title="EcoSort")
app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "app", "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "app", "static", "index.html"))


@app.get("/api/models")
def list_models():
    return {"default": DEFAULT, "device": DEVICE.type, "models": [m["info"] for m in MODELS.values()]}


@app.post("/api/predict")
async def predict(request: Request, model: str = DEFAULT):
    if model not in MODELS:
        raise HTTPException(404, f"unknown model {model}")
    m = MODELS[model]
    try:
        img = Image.open(io.BytesIO(await request.body())).convert("RGB")
    except Exception:
        raise HTTPException(400, "body must be an image")
    t0 = time.perf_counter()
    x = m["tf"](img).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probs = (m["model"](x) / m["temperature"] + m["bias"]).softmax(1)[0].tolist()
    latency = (time.perf_counter() - t0) * 1000
    top = max(range(len(probs)), key=probs.__getitem__)
    return {"label": m["classes"][top], "confidence": probs[top],
            "probs": dict(zip(m["classes"], probs)), "latency_ms": round(latency, 1), "model": model}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
