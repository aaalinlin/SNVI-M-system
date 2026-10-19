"""Native-volume foreground voxel metrics, with explicit case/fold averaging."""
import csv
import fcntl
import json
import statistics
from pathlib import Path
import nibabel as nib
import numpy as np
from polarity_d.runtime import atomic_json, sha256

METRICS = ("dice", "precision", "recall")


def counts_to_metrics(tp, fp, fn):
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    return dict(dice=dice, precision=precision, recall=recall)


def evaluate_fold(predictions, labels, cases, fold, output):
    predictions, labels, output = Path(predictions), Path(labels), Path(output)
    if not cases or len(cases) != len(set(cases)):
        raise ValueError('Expected a nonempty validation case list without duplicates')
    rows = []
    for case in cases:
        pred_img = nib.load(str(predictions / (case + ".nii.gz")))
        true_img = nib.load(str(labels / (case + ".nii.gz")))
        if pred_img.shape != true_img.shape or not np.allclose(pred_img.affine, true_img.affine, atol=1e-4):
            raise ValueError(f"Native geometry mismatch in {case}")
        p, y = np.asarray(pred_img.dataobj), np.asarray(true_img.dataobj)
        if not set(np.unique(p)).issubset({0, 1}) or not set(np.unique(y)).issubset({0, 1}):
            raise ValueError(f"Expected binary foreground labels in {case}")
        p, y = p == 1, y == 1
        tp, fp, fn = int(np.count_nonzero(p & y)), int(np.count_nonzero(p & ~y)), int(np.count_nonzero(~p & y))
        rows.append({"case": case, "fold": int(fold), "tp": tp, "fp": fp, "fn": fn,
                     **counts_to_metrics(tp, fp, fn)})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "case_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"state": "complete", "fold": int(fold), "cases": len(rows),
              "case_macro": {k: statistics.mean(r[k] for r in rows) for k in METRICS},
              "case_std_sample": {k: statistics.stdev(r[k] for r in rows) if len(rows) > 1 else 0 for k in METRICS},
              "voxel_pooled": counts_to_metrics(*(sum(r[k] for r in rows) for k in ["tp", "fp", "fn"])),
              "metric_space": "complete original NIfTI geometry; foreground label 1 only",
              "checkpoint": "checkpoint_final.pth",
              "evaluation_source_sha256": sha256(Path(__file__)),
              "checkpoint_sha256": sha256(output.parent / "checkpoint_final.pth")}
    atomic_json(output / "metrics.json", result)
    return result


def aggregate(model_folder):
    # Different folds can finish together when training on separate GPUs. Hold
    # the lock across both collecting results and replacing the summary so a
    # stale partial snapshot cannot overwrite the final five-fold aggregate.
    model_folder = Path(model_folder)
    with (model_folder / "cross_validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _aggregate_locked(model_folder)


def _aggregate_locked(model_folder):
    model_folder = Path(model_folder)
    completed, all_rows = [], []
    for fold in range(5):
        p = model_folder / f"fold_{fold}" / "evaluation"
        if (p / "metrics.json").exists():
            completed.append(json.loads((p / "metrics.json").read_text()))
            with (p / "case_metrics.csv").open() as f: all_rows.extend(csv.DictReader(f))
    if not completed:
        raise ValueError("No completed folds")
    if len({r["case"] for r in all_rows}) != len(all_rows):
        raise ValueError("A patient was evaluated in multiple folds")
    summary = {"state": "complete" if len(completed) == 5 else "partial", "folds": completed,
               "completed_folds": len(completed), "evaluated_cases": len(all_rows), "independent_test_set": False,
               "fold_mean": {k: statistics.mean(f["case_macro"][k] for f in completed) for k in METRICS},
               "fold_std_sample": {k: statistics.stdev(f["case_macro"][k] for f in completed)
                                   if len(completed) > 1 else None for k in METRICS},
               "oof_case_mean": {k: statistics.mean(float(r[k]) for r in all_rows) for k in METRICS}}
    atomic_json(model_folder / "cross_validation_metrics.json", summary)
    return summary
