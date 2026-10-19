
from __future__ import annotations
import multiprocessing
import os
from typing import List
from pathlib import Path
from warnings import warn

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
from segcore.configuration import default_num_processes


def find_broken_image_and_labels(
    path_to_data_dir: str | Path,
) -> tuple[set[str], set[str]]:

    content = os.listdir(path_to_data_dir)
    unique_ids = [c[:-4] for c in content if c.endswith(".npz")]
    failed_data_ids = set()
    failed_seg_ids = set()
    for unique_id in unique_ids:
        # Try reading data
        try:
            np.load(path_to_data_dir / (unique_id + ".npy"), "r")
        except ValueError:
            failed_data_ids.add(unique_id)
        # Try reading seg
        try:
            np.load(path_to_data_dir / (unique_id + "_seg.npy"), "r")
        except ValueError:
            failed_seg_ids.add(unique_id)

    return failed_data_ids, failed_seg_ids


def try_fix_broken_npy(path_do_data_dir: Path, case_ids: set[str], fix_image: bool):

    for case_id in case_ids:
        for i in range(5):
            try:
                key = "data" if fix_image else "seg"
                suffix = ".npy" if fix_image else "_seg.npy"
                read_npz = np.load(path_do_data_dir / (case_id + ".npz"), "r")[key]
                np.save(path_do_data_dir / (case_id + suffix), read_npz)
                # Try loading the just saved image.
                np.load(path_do_data_dir / (case_id + suffix), "r")
                break
            except ValueError:
                if i == 4:
                    raise ValueError(
                        f"Could not unpack {case_id + suffix} after 5 tries!"
                    )
                continue


def verify_or_stratify_npys(path_to_data_dir: str | Path) -> None:

    path_to_data_dir = Path(path_to_data_dir)

    failed_data_ids, failed_seg_ids = find_broken_image_and_labels(path_to_data_dir)

    if len(failed_data_ids) != 0 or len(failed_seg_ids) != 0:
        warn(
            f"Found {len(failed_data_ids)} faulty data npys and {len(failed_seg_ids)}!\n"
            + f"Faulty images: {failed_data_ids}; Faulty segmentations: {failed_seg_ids})\n"
            + "Trying to fix them now."
        )

        try_fix_broken_npy(path_to_data_dir, failed_data_ids, fix_image=True)
        try_fix_broken_npy(path_to_data_dir, failed_seg_ids, fix_image=False)


def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False) -> None:
    try:
        a = np.load(npz_file)
        if overwrite_existing or not isfile(npz_file[:-3] + "npy"):
            np.save(npz_file[:-3] + "npy", a['data'])
        if unpack_segmentation and (overwrite_existing or not isfile(npz_file[:-4] + "_seg.npy")):
            np.save(npz_file[:-4] + "_seg.npy", a['seg'])
    except KeyboardInterrupt:
        if isfile(npz_file[:-3] + "npy"):
            os.remove(npz_file[:-3] + "npy")
        if isfile(npz_file[:-4] + "_seg.npy"):
            os.remove(npz_file[:-4] + "_seg.npy")
        raise KeyboardInterrupt


def unpack_dataset(folder: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                   num_processes: int = default_num_processes):

    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        npz_files = subfiles(folder, True, None, ".npz", True)
        p.starmap(_convert_to_npy, zip(npz_files,
                                       [unpack_segmentation] * len(npz_files),
                                       [overwrite_existing] * len(npz_files))
                  )


def get_case_identifiers(folder: str) -> List[str]:

    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]
    return case_identifiers


if __name__ == '__main__':
    unpack_dataset('/media/fabian/data/seg_preprocessed/Dataset002_Heart/2d')