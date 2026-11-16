"""Export the trained model to ONNX for the in-browser demo (run by .github/workflows/pages.yml).

    python scripts/export_onnx.py <out_dir>    # writes model.onnx + meta.json
"""
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from models import load_checkpoint  # noqa: E402

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
model, ckpt = load_checkpoint(os.path.join(ROOT, "models", "convnext_tiny.pt"))
n = ckpt["img_size"]
path = os.path.join(out, "model.onnx")
torch.onnx.export(model, torch.randn(1, 3, n, n), path, input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False)

# The checkpoint stores fp16 weights, so ship them as fp16 (half the download) and cast to fp32 when the model loads.
m = onnx.load(path)
casts = []
for init in m.graph.initializer:
    if init.data_type == TensorProto.FLOAT and np.prod(init.dims) > 1024:
        name = init.name
        init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), name + "_fp16"))
        casts.append(helper.make_node("Cast", [name + "_fp16"], [name], to=TensorProto.FLOAT))
nodes = casts + list(m.graph.node)
del m.graph.node[:]
m.graph.node.extend(nodes)
onnx.checker.check_model(m)
onnx.save(m, path)

# The browser model must give the same logits as PyTorch.
x = torch.randn(4, 3, n, n)
with torch.no_grad():
    ref = model(x).numpy()
sess = ort.InferenceSession(path)
got = np.concatenate([sess.run(None, {"input": x[i:i + 1].numpy()})[0] for i in range(len(x))])
assert np.allclose(ref, got, atol=1e-3), f"ONNX output differs from PyTorch by {np.abs(ref - got).max()}"

cal = ckpt.get("calibration", {"temperature": 1.0, "bias": [0.0] * len(ckpt["classes"])})
info = {"name": ckpt["arch"], "val_acc": ckpt["val_acc"], "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "calibrated": "calibration" in ckpt}
metrics_path = os.path.join(ROOT, "results", ckpt["arch"], "metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        t = json.load(f)["test"]
    info.update(test_acc=t["accuracy"], test_f1=t["macro_f1"])
with open(os.path.join(out, "meta.json"), "w") as f:
    json.dump({"classes": ckpt["classes"], "img_size": n, "mean": list(ckpt["mean"]), "std": list(ckpt["std"]),
               "temperature": float(cal["temperature"]), "bias": [float(b) for b in cal["bias"]], "info": info}, f)
print(f"torch {torch.__version__}: exported {path} ({os.path.getsize(path) / 1e6:.1f} MB), max diff vs torch {np.abs(ref - got).max():.2e}")
