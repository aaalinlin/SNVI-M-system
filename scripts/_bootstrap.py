import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vendor")]
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, "2")
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
os.environ.setdefault("nnUNet_def_n_proc", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("nnUNet_compile", "false")
# Preserve existing user options while the backend uses neutral environment names.
for previous, current in [("nnUNet_compile", "SEGCORE_COMPILE"),
                          ("nnUNet_def_n_proc", "SEGCORE_EXPORT_WORKERS"),
                          ("nnUNet_n_proc_DA", "SEGCORE_AUGMENT_WORKERS"),
                          ("nnUNet_keep_files_open", "SEGCORE_KEEP_FILES_OPEN")]:
    if previous in os.environ:
        os.environ.setdefault(current, os.environ[previous])
from polarity_d.runtime import bind_paths
bind_paths()  # Also executes in Python's spawned preprocessing/export workers.
