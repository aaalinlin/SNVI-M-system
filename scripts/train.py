"""Run one held-out fold, native-volume evaluation and model export."""
import _bootstrap
import argparse
import fcntl
import json
import multiprocessing as mp
import os
import subprocess
from pathlib import Path
from polarity_d.runtime import ROOT, atomic_json, configure, environment, sha256, source_manifest, status
from polarity_d.data import workspace_config, raw_inventory, validate_splits

TRAINERS = {"baseline": "nnUNetTrainerPolarityDBaseline", "fixed": "nnUNetTrainerPolarityDFixed",
            "adaptive": "nnUNetTrainerPolarityDAdaptive"}


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["PD_SEED"] = str(args.seed)
    os.environ["PD_EPOCHS"] = str(args.epochs)
    os.environ["PD_DA_WORKERS"] = str(args.workers)
    configure(args.workspace, args.raw)
    w = Path(args.workspace)
    prepared, preprocessed = workspace_config(w)
    if Path(prepared["raw_source"]).resolve() != Path(args.raw).resolve():
        raise ValueError("Use the same raw dataset used for preprocessing")
    if sha256(Path(args.raw) / 'dataset.json') != json.loads((w / 'prepare_request.json').read_text())['dataset_sha256']:
        raise ValueError('Raw dataset.json changed since preprocessing; use a new workspace')
    inventory = raw_inventory(args.raw)
    if inventory != json.loads((w / "raw_verified.json").read_text())["inventory"]:
        raise ValueError("Raw data changed since verification; run prepare.py again")
    verification = json.loads((w / "verification.json").read_text())
    if not verification.get("passed") or not verification.get("production_gpu"):
        raise RuntimeError("Run tests/verify.py --workspace WORKSPACE --gpu GPU --output WORKSPACE/verification.json first")
    if verification.get('configuration_sha256') != prepared['configuration_sha256']:
        raise RuntimeError('GPU verification used a different preparation; rerun tests/verify.py for this workspace')
    for name, expected in verification["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError("Verified implementation changed; rerun tests/verify.py")
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required for the experiment")
    from polarity_d import trainer as trainers
    cls = getattr(trainers, TRAINERS[args.variant])
    plans = json.loads((preprocessed / 'nnUNetPlans.json').read_text())
    dataset = json.loads((preprocessed / 'dataset.json').read_text())
    splits = validate_splits(json.loads((preprocessed / 'splits_final.json').read_text()),
                             [record['case'] for record in prepared['cases']])
    model_folder = w / "nnUNet_results" / prepared['dataset_name'] / (cls.__name__ + "__nnUNetPlans__3d_fullres")
    fold_folder = model_folder / f"fold_{args.fold}"
    fold_folder.mkdir(parents=True, exist_ok=True)
    with (fold_folder / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        provenance_path = fold_folder / "provenance.json"
        critical = {str(p.relative_to(ROOT)): sha256(p) for p in
                    [ROOT / "polarity_d/model.py", ROOT / "polarity_d/trainer.py", ROOT / "scripts/train.py"]}
        protocol = {"variant": args.variant, "fold": args.fold, "seed_base": args.seed,
                    "fold_seed": args.seed + 10000 * args.fold, "epochs": args.epochs,
                    "workers": args.workers, "configuration": "3d_fullres", "lr": 0.01,
                    "optimizer": "SGD(momentum=0.99,nesterov=True,weight_decay=3e-5)",
                    "scheduler": "PolyLR(power=0.9)", "checkpoint_selection": "final",
                    "training_steps_per_epoch": 250, "validation_steps_per_epoch": 50}
        protocol['split_seed'] = prepared['split_seed']
        protocol['configuration_sha256'] = prepared['configuration_sha256']
        if provenance_path.exists():
            previous = json.loads(provenance_path.read_text())
            if not args.resume: raise RuntimeError("Existing run found; use --resume explicitly")
            if previous["protocol"] != protocol or previous["critical_sources"] != critical:
                raise RuntimeError("Source/protocol changed; use a new workspace")
        else:
            if args.resume: raise RuntimeError("No prior run to resume")
            atomic_json(provenance_path, {"protocol": protocol, "critical_sources": critical,
                        "sources": source_manifest(), "environment": environment(),
                        "prepared_manifest_sha256": sha256(w / "prepared.json")})
        if (fold_folder / "complete.json").exists():
            print("Fold already complete.")
            return
        # These private evaluation inputs stay with this user's run, not in the source or exported weight ZIP.
        atomic_json(fold_folder / 'validation_cases.json', splits[args.fold]['val'])
        instance = cls(plans, "3d_fullres", args.fold, dataset, unpack_dataset=False, device=torch.device("cuda"))
        final = fold_folder / "checkpoint_final.pth"
        if final.exists():
            instance.load_checkpoint(str(final))
        else:
            if args.resume:
                options = [fold_folder / n for n in ["checkpoint_latest.pth", "checkpoint_best.pth"]]
                options = [p for p in options if p.exists()]
                if not options: raise RuntimeError("No checkpoint exists; inspect failure before a fresh run")
                chosen = max(options, key=lambda p: torch.load(p, map_location="cpu", weights_only=False)["current_epoch"])
                instance.load_checkpoint(str(chosen))
            status(fold_folder / "status.json", "training_start", **protocol)
            instance.run_training()
        status(fold_folder / "status.json", "full_volume_evaluation", fold=args.fold, completed_epochs=args.epochs)
        instance.perform_actual_validation(save_probabilities=False)
        from polarity_d.evaluation import evaluate_fold, aggregate
        cases = splits[args.fold]['val']
        metrics = evaluate_fold(fold_folder / "validation", Path(args.raw) / "labelsTr", cases,
                                args.fold, fold_folder / "evaluation")
        aggregate(model_folder)
        subprocess.run([os.sys.executable, str(ROOT / "scripts/export_model.py"),
                        "--model-folder", str(model_folder), "--fold", str(args.fold),
                        "--output", str(w / "release" / (args.variant + f"_fold_{args.fold}"))], check=True)
        atomic_json(fold_folder / "complete.json", metrics)
        status(fold_folder / "status.json", "complete", fold=args.fold, metrics=metrics["case_macro"])


if __name__ == "__main__":
    # Workers are recreated after CUDA/CPU thread pools have been used. Forking
    # that process can deadlock on inherited locks; spawn fresh interpreters.
    mp.set_start_method("spawn", force=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--variant", choices=TRAINERS, default="adaptive")
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--gpu", default="4", help="Physical GPU index or GPU UUID")
    p.add_argument("--epochs", type=int, default=700)
    p.add_argument("--seed", type=int, default=20261017)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.epochs < 1 or a.workers < 1 or not 0 <= a.seed < 2**32 - 10000*4 - 1000*a.epochs - 1000000 - a.workers:
        p.error("Epochs/workers must be positive and seeds must fit NumPy uint32")
    try:
        run(a)
    except Exception as error:
        status(Path(a.workspace) / f"failure_{a.variant}_fold_{a.fold}.json", "failed",
               fold=a.fold, variant=a.variant, error=repr(error))
        raise
