"""Plan and preprocess a user's MRI data; keep derived metadata in the workspace."""
import _bootstrap
import argparse
import fcntl
import json
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from polarity_d.runtime import ROOT, atomic_json, configure, sha256, status
from polarity_d.data import dataset_info, make_splits, validate_splits, raw_inventory


def one_case(case):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from segcore.utilities.plans_handling.plans_handler import PlansManager
    raw = Path(os.environ["PD_RAW_DATASET"])
    workspace = Path(os.environ["PD_WORKSPACE"])
    dst = workspace / "nnUNet_preprocessed" / raw.name
    plans = PlansManager(str(dst / "nnUNetPlans.json"))
    config = plans.get_configuration("3d_fullres")
    dataset = json.loads((dst / "dataset.json").read_text())
    image, label = raw / "imagesTr" / (case + "_0000.nii.gz"), raw / "labelsTr" / (case + ".nii.gz")
    prefix = dst / config.data_identifier / case
    receipt = workspace / "case_manifests" / (case + ".json")
    hashes = {"image_sha256": sha256(image), "label_sha256": sha256(label),
              "plans_sha256": sha256(dst / "nnUNetPlans.json")}
    previous = json.loads(receipt.read_text()) if receipt.exists() else {}
    if (not all(previous.get(k) == v for k, v in hashes.items()) or
            not prefix.with_suffix(".npz").exists() or not prefix.with_suffix(".pkl").exists()):
        config.preprocessor_class(verbose=False).run_case_save(
            str(prefix), [str(image)], str(label), plans, config, dataset)
    with np.load(prefix.with_suffix(".npz")) as a:
        data, seg = a["data"], a["seg"]
        assert np.isfinite(data).all(), case
        assert set(np.unique(seg)).issubset({-1, 0, 1}), case
        shape = list(data.shape)
        # Write mmap arrays atomically once; avoid a large unpack burst at train startup.
        for suffix, array in [(".npy", data), ("_seg.npy", seg)]:
            output = prefix.parent / (case + suffix)
            if not output.exists() or previous.get("image_sha256") != hashes["image_sha256"] or previous.get("label_sha256") != hashes["label_sha256"] or previous.get("plans_sha256") != hashes["plans_sha256"]:
                temporary = output.with_name(output.name + ".tmp")
                with temporary.open("wb") as f:
                    np.save(f, array)
                temporary.replace(output)
    record = {"case": case, "shape": shape, **hashes, "preprocessed_sha256": sha256(prefix.with_suffix(".npz"))}
    atomic_json(receipt, record)
    return record


def prepare(args):
    configure(args.workspace, args.raw)
    w, raw = Path(args.workspace).resolve(), Path(args.raw).resolve()
    if w.is_relative_to(ROOT) or w.is_relative_to(raw) or raw.is_relative_to(w):
        raise ValueError('Keep the workspace outside the source and raw-data directories')
    dataset, cases = dataset_info(raw)
    groups = json.loads(Path(args.groups).read_text()) if args.groups else None
    if args.splits and groups:
        raise ValueError('Use either --splits or --groups, not both')
    splits = (validate_splits(json.loads(Path(args.splits).read_text()), cases) if args.splits
              else make_splits(cases, args.seed, groups))
    if groups:
        for split in splits:
            if {groups[c] for c in split['train']} & {groups[c] for c in split['val']}:
                raise ValueError('A patient group occurs in both training and validation')
    w.mkdir(parents=True, exist_ok=True)
    with (w / "prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request = {'raw_source': str(raw), 'dataset_sha256': sha256(raw / 'dataset.json'),
                   'inventory': raw_inventory(raw), 'split_seed': args.seed,
                   'gpu_memory_target_gb': args.gpu_memory_target,
                   'provided_plans_sha256': sha256(args.plans) if args.plans else None,
                   'splits': splits}
        request_path = w / 'prepare_request.json'
        if request_path.exists() and json.loads(request_path.read_text()) != request:
            raise ValueError('Raw data or preparation settings changed; use a new workspace')
        atomic_json(request_path, request)
        if (w / 'prepared.json').exists():
            from polarity_d.data import workspace_config
            workspace_config(w)  # Reject edits to an already prepared experiment.
        dst = w / "nnUNet_preprocessed" / raw.name
        dst.mkdir(parents=True, exist_ok=True)
        verified = w / "raw_verified.json"
        if not verified.exists():
            status(w / "prepare_status.json", "verifying_raw", cases=len(cases))
            from segcore.experiment_planning.verify_dataset_integrity import verify_dataset_integrity
            verify_dataset_integrity(str(raw), num_processes=args.workers)
            atomic_json(verified, {"passed": True, "inventory": request['inventory']})
        from segcore.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
        extractor = DatasetFingerprintExtractor(raw.name, args.workers)
        # MRI normalization is per image; cohort intensity statistics are not used.
        extractor.num_foreground_voxels_for_intensitystats = min(10_000_000, len(cases) * 10000)
        extractor.run(overwrite_existing=False)
        plan_file = dst / 'nnUNetPlans.json'
        if not plan_file.exists():
            if args.plans:
                plans = json.loads(Path(args.plans).read_text())
                plans['dataset_name'] = raw.name
                plans['plans_name'] = 'nnUNetPlans'
                from segcore.utilities.plans_handling.plans_handler import PlansManager
                fullres = dict(PlansManager(plans).get_configuration('3d_fullres').configuration)
                fullres['data_identifier'] = 'nnUNetPlans_3d_fullres'
                plans['configurations'] = {'3d_fullres': fullres}
                atomic_json(plan_file, plans)
            else:
                status(w / 'prepare_status.json', 'planning')
                from segcore.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
                plans = ExperimentPlanner(raw.name, gpu_memory_target_in_gb=args.gpu_memory_target,
                                          plans_name='nnUNetPlans').plan_experiment()
                if '3d_fullres' not in plans['configurations']:
                    raise ValueError('This model requires 3D volumes, not single-slice images')
                plans['configurations'] = {'3d_fullres': plans['configurations']['3d_fullres']}
                atomic_json(plan_file, plans)
        plans = json.loads(plan_file.read_text())
        from segcore.utilities.plans_handling.plans_handler import PlansManager
        config = PlansManager(plans).get_configuration('3d_fullres')
        if (config.UNet_class_name != 'PlainConvUNet' or len(config.patch_size) != 3
                or len(config.conv_kernel_sizes) < 3 or config.previous_stage_name is not None):
            raise ValueError('Use a non-cascaded 3D PlainConvUNet with at least three stages')
        if config.normalization_schemes != ['ZScoreNormalization']:
            raise ValueError('The MRI plan must use ZScoreNormalization')
        if config.batch_size < 1:
            raise ValueError('The training plan must include a positive batch_size')
        atomic_json(dst / 'splits_final.json', splits)
        atomic_json(dst / 'dataset.json', dataset)
        (dst / config.data_identifier).mkdir(exist_ok=True)
        (dst / 'gt_segmentations').mkdir(exist_ok=True)
        for case in cases:
            shutil.copy2(raw / "labelsTr" / (case + ".nii.gz"), dst / "gt_segmentations" / (case + ".nii.gz"))
        records = []
        status(w / "prepare_status.json", "preprocessing", completed=0, total=len(cases))
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
            futures = [ex.submit(one_case, case) for case in cases]
            for f in as_completed(futures):
                records.append(f.result())
                if len(records) % 10 == 0 or len(records) == len(cases):
                    status(w / "prepare_status.json", "preprocessing", completed=len(records), total=len(cases))
        atomic_json(w / "prepared.json", {"raw_source": str(raw.resolve()), "cases": sorted(records, key=lambda x: x["case"]),
                    'dataset_name': raw.name, 'split_seed': args.seed,
                    "configuration": "3d_fullres", "case_count": len(cases),
                    'configuration_sha256': {name: sha256(dst / name) for name in
                        ['nnUNetPlans.json', 'splits_final.json', 'dataset.json', 'dataset_fingerprint.json']}})
        status(w / "prepare_status.json", "complete", completed=len(cases), total=len(cases))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument('--seed', type=int, default=20261017, help='Seed for the five-fold split')
    p.add_argument('--gpu-memory-target', type=float, default=8, help='Planner memory target in GiB; verify on the target GPU')
    p.add_argument('--plans', help='Optional compatible training plans instead of automatic planning')
    p.add_argument('--splits', help='Optional private five-fold splits JSON')
    p.add_argument('--groups', help='Optional JSON mapping case identifiers to patient identifiers')
    a = p.parse_args()
    if a.workers < 1 or a.gpu_memory_target <= 0 or not 0 <= a.seed < 2**32:
        p.error('Workers/memory must be positive and the seed must fit uint32')
    try:
        prepare(a)
    except Exception as error:
        status(Path(a.workspace) / "prepare_status.json", "failed", error=repr(error))
        raise
