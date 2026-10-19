
from typing import Union

from segcore.paths import seg_preprocessed, seg_raw, seg_results
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np


def find_candidate_datasets(dataset_id: int):
    startswith = "Dataset%03.0d" % dataset_id
    if seg_preprocessed is not None and isdir(seg_preprocessed):
        candidates_preprocessed = subdirs(seg_preprocessed, prefix=startswith, join=False)
    else:
        candidates_preprocessed = []

    if seg_raw is not None and isdir(seg_raw):
        candidates_raw = subdirs(seg_raw, prefix=startswith, join=False)
    else:
        candidates_raw = []

    candidates_trained_models = []
    if seg_results is not None and isdir(seg_results):
        candidates_trained_models += subdirs(seg_results, prefix=startswith, join=False)

    all_candidates = candidates_preprocessed + candidates_raw + candidates_trained_models
    unique_candidates = np.unique(all_candidates)
    return unique_candidates


def convert_id_to_dataset_name(dataset_id: int):
    unique_candidates = find_candidate_datasets(dataset_id)
    if len(unique_candidates) > 1:
        raise RuntimeError("More than one dataset name found for dataset id %d. Please correct that. (I looked in the "
                           "following folders:\n%s\n%s\n%s" % (dataset_id, seg_raw, seg_preprocessed, seg_results))
    if len(unique_candidates) == 0:
        raise RuntimeError(f"Could not find a dataset with the ID {dataset_id}. Make sure the requested dataset ID "
                           f"exists and that nnU-Net knows where raw and preprocessed data are located "
                           f"(see Documentation - Installation). Here are your currently defined folders:\n"
                           f"seg_preprocessed={os.environ.get('seg_preprocessed') if os.environ.get('seg_preprocessed') is not None else 'None'}\n"
                           f"seg_results={os.environ.get('seg_results') if os.environ.get('seg_results') is not None else 'None'}\n"
                           f"seg_raw={os.environ.get('seg_raw') if os.environ.get('seg_raw') is not None else 'None'}\n"
                           f"If something is not right, adapt your environment variables.")
    return unique_candidates[0]


def convert_dataset_name_to_id(dataset_name: str):
    assert dataset_name.startswith("Dataset")
    dataset_id = int(dataset_name[7:10])
    return dataset_id


def maybe_convert_to_dataset_name(dataset_name_or_id: Union[int, str]) -> str:
    if isinstance(dataset_name_or_id, str) and dataset_name_or_id.startswith("Dataset"):
        return dataset_name_or_id
    if isinstance(dataset_name_or_id, str):
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError("dataset_name_or_id was a string and did not start with 'Dataset' so we tried to "
                             "convert it to a dataset ID (int). That failed, however. Please give an integer number "
                             "('1', '2', etc) or a correct dataset name. Your input: %s" % dataset_name_or_id)
    return convert_id_to_dataset_name(dataset_name_or_id)
