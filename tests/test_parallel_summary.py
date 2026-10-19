"""Concurrent fold summaries must not collide or leave a partial result."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _bootstrap
import csv
import json
import subprocess
import tempfile
from polarity_d.runtime import ROOT, atomic_json


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="pd_parallel_summary_") as temporary:
        model = Path(temporary)
        for fold in range(5):
            folder = model / f"fold_{fold}" / "evaluation"
            atomic_json(folder / "metrics.json", {"state": "complete", "fold": fold,
                "case_macro": {"dice": 0.8, "precision": 0.75, "recall": 0.9}, "cases": 1})
            with (folder / "case_metrics.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["case", "fold", "dice", "precision", "recall"])
                writer.writeheader()
                writer.writerow(dict(case=f"synthetic_{fold}", fold=fold, dice=0.8, precision=0.75, recall=0.9))
        code = ("import sys;sys.path.insert(0,sys.argv[1]);import _bootstrap;"
                "from polarity_d.evaluation import aggregate;"
                "[aggregate(sys.argv[2]) for _ in range(10)]")
        children = [subprocess.Popen([sys.executable, "-B", "-c", code, str(ROOT / "scripts"), str(model)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    for _ in range(4)]
        for child in children:
            out, error = child.communicate(timeout=60)
            assert child.returncode == 0, error
        result = json.loads((model / "cross_validation_metrics.json").read_text())
        assert result["state"] == "complete" and result["completed_folds"] == 5
        assert result["evaluated_cases"] == 5
        assert result["fold_mean"] == {"dice": 0.8, "precision": 0.75, "recall": 0.9}
        assert result["fold_std_sample"] == {k: 0.0 for k in ["dice", "precision", "recall"]}
    print("PASS: 4 concurrent writers, 40 aggregate updates, correct five-fold summary.")
