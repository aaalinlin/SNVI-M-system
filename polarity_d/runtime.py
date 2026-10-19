"""Explicit paths, atomic metadata, and source provenance for reproducible runs."""
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def configure(workspace, raw):
    os.environ["PD_WORKSPACE"] = str(Path(workspace).resolve())
    os.environ["PD_RAW_DATASET"] = str(Path(raw).resolve())
    bind_paths()


def bind_paths():
    if "PD_WORKSPACE" not in os.environ:
        return
    import segcore.paths as paths
    w = Path(os.environ["PD_WORKSPACE"])
    raw = Path(os.environ["PD_RAW_DATASET"])
    paths.seg_raw = str(raw.parent)
    paths.seg_preprocessed = str(w / "nnUNet_preprocessed")
    paths.seg_results = str(w / "nnUNet_results")
    for previous, current in [("nnUNet_raw", "seg_raw"),
                              ("nnUNet_preprocessed", "seg_preprocessed"),
                              ("nnUNet_results", "seg_results")]:
        os.environ[previous] = getattr(paths, current)
        os.environ[current] = getattr(paths, current)


def source_manifest():
    files = []
    for sub in ["polarity_d", "scripts", "tests", "configs", "vendor"]:
        for p in sorted((ROOT / sub).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                files.append({"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)})
    return files


def environment():
    import torch
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pip_freeze": subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True).splitlines()}


def status(path, stage, **kwargs):
    value = {"stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(),
             "pid": os.getpid(), **kwargs}
    atomic_json(path, value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
