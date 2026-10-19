import _bootstrap
import argparse
import json
import zipfile
from pathlib import Path
from polarity_d.runtime import atomic_json, sha256
from polarity_d.export import inference_metadata

def export_model(model_folder, fold, output):
    import torch
    model_folder, output = Path(model_folder).resolve(), Path(output).resolve()
    if output.is_relative_to(model_folder) or model_folder.is_relative_to(output):
        raise ValueError('Export to a separate directory outside the trained model folder')
    names = {'plans.json', 'dataset.json', f'fold_{fold}/checkpoint_final.pth'}
    if output.exists() and any(p.relative_to(output).as_posix() not in names
                               for p in output.rglob('*') if p.is_file()):
        raise ValueError('Export directory contains other files; choose a new empty directory')
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(model_folder / f'fold_{fold}/checkpoint_final.pth', map_location='cpu', weights_only=False)
    configuration = checkpoint['init_args']['configuration']
    plans, dataset = inference_metadata(json.loads((model_folder / 'plans.json').read_text()),
                                        json.loads((model_folder / 'dataset.json').read_text()), configuration)
    atomic_json(output / 'plans.json', plans)
    atomic_json(output / 'dataset.json', dataset)
    target = output / f"fold_{fold}"
    target.mkdir(exist_ok=True)
    torch.save({'network_weights': checkpoint['network_weights'], 'trainer_name': checkpoint['trainer_name'],
                'init_args': {'configuration': configuration},
                'inference_allowed_mirroring_axes': checkpoint.get('inference_allowed_mirroring_axes')},
               target / 'checkpoint_final.pth')
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in sorted(names):
            z.write(output / name, name)
    atomic_json(archive.with_suffix(".sha256.json"), {"filename": archive.name, "sha256": sha256(archive)})
    print(str(archive), flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export prediction-only weights and configuration without patient lists or training records")
    p.add_argument("--model-folder", required=True)
    p.add_argument("--fold", type=int, required=True, choices=range(5))
    p.add_argument("--output", required=True)
    a = p.parse_args()
    export_model(a.model_folder, a.fold, a.output)
