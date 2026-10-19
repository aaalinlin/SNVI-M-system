"""Validate user-owned MRI datasets and deterministic, patient-level splits."""
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold


def dataset_info(raw):
    raw = Path(raw).resolve()
    if not raw.name.startswith('Dataset'):
        raise ValueError('The raw folder name must start with Dataset (for example Dataset or Dataset001_MRI)')
    dataset = json.loads((raw / 'dataset.json').read_text())
    if dataset.get('file_ending') != '.nii.gz':
        raise ValueError('This pipeline expects .nii.gz images and labels')
    channels = dataset.get('channel_names', {})
    if set(channels) != {'0'}:
        raise ValueError('This pipeline supports one MRI input channel, named CASE_0000.nii.gz')
    from segcore.preprocessing.normalization.map_channel_name_to_normalization import get_normalization_scheme
    if get_normalization_scheme(channels['0']).__name__ != 'ZScoreNormalization':
        raise ValueError('This MRI pipeline requires per-image ZScoreNormalization')
    labels = dataset.get('labels', {})
    if labels.get('background') != 0 or len(labels) != 2 or sorted(labels.values()) != [0, 1]:
        raise ValueError('Use binary labels: background=0 and one foreground class=1')
    if 'dataset' in dataset or 'regions_class_order' in dataset:
        raise ValueError('Use imagesTr/labelsTr folders and ordinary binary labels')
    cases = sorted(p.name[:-7] for p in (raw / 'labelsTr').glob('*.nii.gz'))
    images = sorted(p.name[:-12] for p in (raw / 'imagesTr').glob('*_0000.nii.gz'))
    if cases != images or len(list((raw / 'imagesTr').glob('*.nii.gz'))) != len(images):
        raise ValueError('Each CASE.nii.gz label must have exactly one CASE_0000.nii.gz image')
    if len(cases) < 5:
        raise ValueError('Five-fold cross-validation needs at least five independent cases')
    if dataset.get('numTraining') != len(cases):
        raise ValueError(f'Set numTraining in the raw dataset.json to the actual case count: {len(cases)}')
    return dataset, cases


def validate_splits(splits, cases):
    expected, heldout = set(cases), []
    if len(splits) != 5:
        raise ValueError('Exactly five folds are required')
    for split in splits:
        train, val = split['train'], split['val']
        if (not train or not val or len(set(train)) != len(train) or len(set(val)) != len(val)
                or set(train) & set(val) or set(train) | set(val) != expected):
            raise ValueError('Every fold must partition all cases without overlap or duplicates')
        heldout.extend(val)
    if sorted(heldout) != sorted(cases):
        raise ValueError('Each case must occur in exactly one validation fold')
    return splits


def make_splits(cases, seed, groups=None):
    """Optionally keep multiple scans from the same patient in a single fold."""
    cases = sorted(cases)
    if groups is None:
        groups = {case: case for case in cases}
    if set(groups) != set(cases) or not all(isinstance(g, str) and g for g in groups.values()):
        raise ValueError('Groups JSON must map every case identifier to a nonempty patient/group identifier')
    units = np.asarray(sorted(set(groups.values())))
    if len(units) < 5:
        raise ValueError('At least five independent patient groups are needed')
    splits = []
    for train_idx, val_idx in KFold(5, shuffle=True, random_state=seed).split(units):
        train_units, val_units = set(units[train_idx]), set(units[val_idx])
        splits.append({'train': [c for c in cases if groups[c] in train_units],
                       'val': [c for c in cases if groups[c] in val_units]})
    return validate_splits(splits, cases)


def raw_inventory(raw):
    raw = Path(raw)
    return [{'path': str(p.relative_to(raw)), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
            for sub in ['imagesTr', 'labelsTr'] for p in sorted((raw / sub).glob('*.nii.gz'))]


def workspace_config(workspace):
    workspace = Path(workspace)
    prepared = json.loads((workspace / 'prepared.json').read_text())
    folder = workspace / 'nnUNet_preprocessed' / prepared['dataset_name']
    from polarity_d.runtime import sha256
    for name, expected in prepared['configuration_sha256'].items():
        if sha256(folder / name) != expected:
            raise ValueError(f'Prepared configuration changed: {name}. Use a new workspace for a new experiment.')
    return prepared, folder
