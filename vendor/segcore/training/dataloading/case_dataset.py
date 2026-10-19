
import os
from typing import List

import numpy as np
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
from segcore.training.dataloading.utils import get_case_identifiers


class SegDataset(object):
    def __init__(self, folder: str, case_identifiers: List[str] = None,
                 num_images_properties_loading_threshold: int = 0,
                 folder_with_segs_from_previous_stage: str = None):

        super().__init__()
        # print('loading dataset')
        if case_identifiers is None:
            case_identifiers = get_case_identifiers(folder)
        case_identifiers.sort()

        self.dataset = {}
        for c in case_identifiers:
            self.dataset[c] = {}
            self.dataset[c]['data_file'] = join(folder, f"{c}.npz")
            self.dataset[c]['properties_file'] = join(folder, f"{c}.pkl")
            if folder_with_segs_from_previous_stage is not None:
                self.dataset[c]['seg_from_prev_stage_file'] = join(folder_with_segs_from_previous_stage, f"{c}.npz")

        if len(case_identifiers) <= num_images_properties_loading_threshold:
            for i in self.dataset.keys():
                self.dataset[i]['properties'] = load_pickle(self.dataset[i]['properties_file'])

        self.keep_files_open = ('SEGCORE_KEEP_FILES_OPEN' in os.environ.keys()) and \
                               (os.environ['SEGCORE_KEEP_FILES_OPEN'].lower() in ('true', '1', 't'))
        # print(f'SegDataset.keep_files_open: {self.keep_files_open}')

    def __getitem__(self, key):
        ret = {**self.dataset[key]}
        if 'properties' not in ret.keys():
            ret['properties'] = load_pickle(ret['properties_file'])
        return ret

    def __setitem__(self, key, value):
        return self.dataset.__setitem__(key, value)

    def keys(self):
        return self.dataset.keys()

    def __len__(self):
        return self.dataset.__len__()

    def items(self):
        return self.dataset.items()

    def values(self):
        return self.dataset.values()

    def load_case(self, key):
        entry = self[key]
        if 'open_data_file' in entry.keys():
            data = entry['open_data_file']
            # print('using open data file')
        elif isfile(entry['data_file'][:-4] + ".npy"):
            data = np.load(entry['data_file'][:-4] + ".npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_data_file'] = data
                # print('saving open data file')
        else:
            data = np.load(entry['data_file'])['data']

        if 'open_seg_file' in entry.keys():
            seg = entry['open_seg_file']
            # print('using open data file')
        elif isfile(entry['data_file'][:-4] + "_seg.npy"):
            seg = np.load(entry['data_file'][:-4] + "_seg.npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_seg_file'] = seg
                # print('saving open seg file')
        else:
            seg = np.load(entry['data_file'])['seg']

        if 'seg_from_prev_stage_file' in entry.keys():
            if isfile(entry['seg_from_prev_stage_file'][:-4] + ".npy"):
                seg_prev = np.load(entry['seg_from_prev_stage_file'][:-4] + ".npy", 'r')
            else:
                seg_prev = np.load(entry['seg_from_prev_stage_file'])['seg']
            seg = np.vstack((seg, seg_prev[None]))

        return data, seg, entry['properties']


if __name__ == '__main__':


    folder = '/media/fabian/data/seg_preprocessed/Dataset003_Liver/3d_lowres'
    ds = SegDataset(folder, num_images_properties_loading_threshold=0) # this should not load the properties!

    ks = ds['liver_0'].keys()
    assert 'properties' in ks

    ds = SegDataset(folder, num_images_properties_loading_threshold=1000)

    shutil.move(join(folder, 'liver_0.pkl'), join(folder, 'liver_XXX.pkl'))

    ks = ds['liver_0'].keys()
    assert 'properties' in ks

    shutil.move(join(folder, 'liver_XXX.pkl'), join(folder, 'liver_0.pkl'))


    ds = SegDataset(folder, num_images_properties_loading_threshold=0)

    shutil.move(join(folder, 'liver_0.pkl'), join(folder, 'liver_XXX.pkl'))

    try:
        ks = ds['liver_0'].keys()
        raise RuntimeError('we should not have come here')
    except FileNotFoundError:
        print('all good')
        # move file back
        shutil.move(join(folder, 'liver_XXX.pkl'), join(folder, 'liver_0.pkl'))

