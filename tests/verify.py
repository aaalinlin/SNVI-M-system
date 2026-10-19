"""Behavioral checks, checkpoint reload, and optional production-shape GPU check."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _bootstrap
import argparse
import copy
import json
import os
import tempfile
import time
import numpy as np
import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from polarity_d.model import DState, route_packet, PolarityDUNet3D
from polarity_d.evaluation import counts_to_metrics, evaluate_fold
from polarity_d.runtime import ROOT, atomic_json, configure, sha256
from fixtures import synthetic_plans, synthetic_dataset


def tiny_network():
    return PlainConvUNet(1, 3, [4, 8, 16], nn.Conv3d, [[3]*3]*3,
                         [[1]*3, [2]*3, [2]*3], [2]*3, 2, [2]*2,
                         conv_bias=True, norm_op=nn.InstanceNorm3d,
                         norm_op_kwargs={"eps": 1e-5, "affine": True},
                         nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True}, deep_supervision=True)


def test_routes():
    # Negative feature values stay negative; the metadata sign controls routing.
    payload = torch.tensor([-3., 0., 4.], requires_grad=True)
    for sign in [-1, 0, 1]:
        state = DState()
        packet = route_packet(payload, sign, state)
        assert packet.disposable is payload and state.stored is payload and not state.open
        assert (packet.positive is payload) == (sign == 1)
        assert (packet.negative is payload) == (sign == -1)
        second = route_packet(payload * 2, sign, state)
        assert second.disposable is None and state.stored is payload
        reset = route_packet(payload * 3, sign, state, reset=True)
        assert reset.disposable is state.stored
        torch.testing.assert_close(state.stored, payload * 3)
    state = DState()
    a = route_packet(payload, -1, state)
    (a.disposable.sum() + a.negative.sum()).backward()
    torch.testing.assert_close(payload.grad, torch.full_like(payload, 2))


def test_equivalence_and_isolation():
    torch.manual_seed(42)
    baseline = tiny_network()
    fixed = PolarityDUNet3D(copy.deepcopy(baseline))
    x1 = torch.randn(2, 1, 16, 16, 16, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    a, b = baseline(x1), fixed(x2)
    for aa, bb in zip(a, b): torch.testing.assert_close(aa, bb, rtol=0, atol=0)
    sum(z.square().mean() for z in a).backward()
    sum(z.square().mean() for z in b).backward()
    torch.testing.assert_close(x1.grad, x2.grad, rtol=0, atol=0)
    original_params, routed_params = dict(baseline.named_parameters()), dict(fixed.named_parameters())
    assert set(original_params) == set(routed_params)
    for name in original_params:
        torch.testing.assert_close(original_params[name].grad, routed_params[name].grad, rtol=0, atol=0)
    adaptive = PolarityDUNet3D(copy.deepcopy(baseline), adaptive=True)
    adaptive.eval()
    adaptive.decoder.deep_supervision = False
    sample = torch.randn(2, 1, 16, 16, 16)
    first = adaptive(sample)
    adaptive(torch.randn_like(sample))
    torch.testing.assert_close(adaptive(sample), first, rtol=0, atol=0)
    torch.testing.assert_close(adaptive(sample.flip(0)).flip(0), first)
    adaptive.zero_grad(set_to_none=True)
    adaptive(sample).square().mean().backward()
    for reader in adaptive.decoder.readers:
        assert reader.selector.weight.grad is not None
        assert torch.isfinite(reader.selector.weight.grad).all()
        assert reader.selector.weight.grad.abs().sum() > 0
    return sum(p.numel() for p in adaptive.parameters()) - sum(p.numel() for p in baseline.parameters())


def small_plans():
    return synthetic_plans()


def test_checkpoint_and_predictor(tmp):
    configure(tmp, tmp / "DatasetSynthetic")
    from polarity_d.trainer import nnUNetTrainerPolarityDAdaptive
    dataset = synthetic_dataset()
    plans = small_plans()
    a = nnUNetTrainerPolarityDAdaptive(plans, "3d_fullres", 0, dataset, False, torch.device("cpu"))
    a.initialize()
    x = torch.randn(2, 1, 16, 16, 16)
    y = torch.randint(0, 2, (2, 1, 16, 16, 16))
    batch = {"data": x, "target": [y, y[:, :, ::2, ::2, ::2].contiguous()]}
    assert np.isfinite(a.train_step(batch)["loss"])
    ck = Path(a.output_folder) / "checkpoint_final.pth"
    a.save_checkpoint(str(ck))
    b = nnUNetTrainerPolarityDAdaptive(plans, "3d_fullres", 0, dataset, False, torch.device("cpu"))
    b.load_checkpoint(str(ck))
    assert b.current_epoch == 1
    a.train_step(batch)
    b.train_step(batch)
    for pa, pb in zip(a.network.parameters(), b.network.parameters()):
        torch.testing.assert_close(pa, pb, rtol=0, atol=0)
    model_folder = ck.parent.parent
    atomic_json(model_folder / "plans.json", plans)
    atomic_json(model_folder / "dataset.json", dataset)
    from segcore.inference.predict_from_raw_data import SegPredictor
    predictor = SegPredictor(device=torch.device("cpu"), perform_everything_on_device=False,
                               use_mirroring=False, use_gaussian=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model_folder), [0], "checkpoint_final.pth")
    saved = torch.load(ck, map_location="cpu", weights_only=False)
    a.network.load_state_dict(saved["network_weights"])
    predictor.network.load_state_dict(saved["network_weights"])
    a.network.decoder.deep_supervision = False
    a.network.eval()
    predictor.network.eval()
    with torch.no_grad(): torch.testing.assert_close(predictor.network(x), a.network(x), rtol=0, atol=0)
    # Exercise actual sliding-window inference on a synthetic volume with overlaps.
    logits = predictor.predict_sliding_window_return_logits(torch.randn(1, 20, 20, 20))
    assert logits.shape == (2, 20, 20, 20) and torch.isfinite(logits).all()


def test_metrics(tmp):
    assert counts_to_metrics(2, 1, 2) == {"dice": 4/7, "precision": 2/3, "recall": .5}
    assert counts_to_metrics(0, 0, 0) == {k: 1.0 for k in ["dice", "precision", "recall"]}
    assert counts_to_metrics(0, 0, 3) == {k: 0.0 for k in ["dice", "precision", "recall"]}
    import nibabel as nib
    pred, labels = tmp / "pred", tmp / "labels"
    pred.mkdir(); labels.mkdir()
    y = np.zeros((8, 8, 8), dtype=np.uint8); y[2:5, 2:5, 2:5] = 1
    aff = np.diag([.5, .5, 1., 1.])
    for path in [pred / "001.nii.gz", labels / "001.nii.gz"]:
        nib.save(nib.Nifti1Image(y, aff), path)
    (tmp / "checkpoint_final.pth").write_bytes(b"synthetic-test-only")
    r = evaluate_fold(pred, labels, ["001"], 0, tmp / "evaluation")
    assert r["case_macro"] == {k: 1.0 for k in ["dice", "precision", "recall"]}


def test_gpu(gpu, tmp, config_folder):
    configure(tmp, tmp / "DatasetSynthetic")
    from polarity_d.trainer import nnUNetTrainerPolarityDAdaptive
    plans = json.loads((config_folder / 'nnUNetPlans.json').read_text())
    plans['dataset_name'] = 'DatasetSynthetic'
    dataset = json.loads((config_folder / 'dataset.json').read_text())
    t = nnUNetTrainerPolarityDAdaptive(plans, "3d_fullres", 0, dataset, False, torch.device("cuda"))
    t.initialize()
    shape = t.configuration_manager.patch_size
    batch_size = t.configuration_manager.batch_size
    x = torch.randn(batch_size, 1, *shape)
    y = torch.zeros((batch_size, 1, *shape), dtype=torch.int64)
    y[(slice(None), slice(None), *(slice(s//4, 3*s//4) for s in shape))] = 1
    targets = [torch.nn.functional.interpolate(y.float(), size=[int(round(a*b)) for a, b in zip(shape, scale)], mode="nearest").long()
               for scale in t._get_deep_supervision_scales()]
    batch = {"data": x, "target": targets}
    start = time.perf_counter()
    losses = [float(t.train_step(batch)["loss"]) for _ in range(2)]
    assert all(np.isfinite(losses))
    for reader in t.network.decoder.readers:
        assert reader.selector.weight.grad is not None and torch.isfinite(reader.selector.weight.grad).all()
    t.network.eval()
    with torch.no_grad(): val = t.validation_step(batch)
    assert np.isfinite(val["loss"])
    torch.cuda.synchronize()
    return {"gpu": gpu, "shape": list(shape), "batch": batch_size, "losses": losses,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "seconds": time.perf_counter()-start,
            "parameters": sum(p.numel() for p in t.network.parameters())}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", default=None)
    p.add_argument("--output", required=True)
    p.add_argument('--workspace', help='Prepared workspace whose actual training shape will be checked on the GPU')
    args = p.parse_args()
    prepared, config_folder = None, None
    if args.workspace:
        from polarity_d.data import workspace_config
        prepared, config_folder = workspace_config(args.workspace)
    if args.gpu is not None and config_folder is None:
        p.error('--gpu requires --workspace pointing to successfully prepared data')
    if args.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_num_threads(2)
    report = {"passed": False}
    with tempfile.TemporaryDirectory(prefix="polarity_d_verify_") as td:
        tmp = Path(td)
        test_routes(); report["routes"] = True
        report["tiny_adaptive_extra_parameters"] = test_equivalence_and_isolation()
        report["fixed_forward_backward_equivalence"] = True
        report["state_isolation_and_gate_gradients"] = True
        test_checkpoint_and_predictor(tmp); report["checkpoint_resume_and_sliding_window"] = True
        test_metrics(tmp); report["native_geometry_and_metrics"] = True
        if args.gpu is not None: report["production_gpu"] = test_gpu(args.gpu, tmp, config_folder)
    report["passed"] = True
    report['configuration_sha256'] = prepared['configuration_sha256'] if prepared else None
    report["source_sha256"] = {name: sha256(ROOT / name) for name in
        ['polarity_d/model.py', 'polarity_d/trainer.py', 'polarity_d/data.py', 'polarity_d/export.py',
         'scripts/prepare.py', 'scripts/train.py']}
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
