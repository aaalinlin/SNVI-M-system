

import os
join = os.path.join
"""
Please make sure your data is organized as follows:

data/
├── seg_raw/
│   ├── Dataset701_AbdomenCT/
│   │   ├── imagesTr
│   │   │   ├── FLARE22_Tr_0001_0000.nii.gz
│   │   │   ├── FLARE22_Tr_0002_0000.nii.gz
│   │   │   ├── ...
│   │   ├── labelsTr
│   │   │   ├── FLARE22_Tr_0001.nii.gz
│   │   │   ├── FLARE22_Tr_0002.nii.gz
│   │   │   ├── ...
│   │   ├── dataset.json
│   ├── Dataset702_AbdomenMR/
│   │   ├── imagesTr
│   │   │   ├── amos_0507_0000.nii.gz
│   │   │   ├── amos_0508_0000.nii.gz
│   │   │   ├── ...
│   │   ├── labelsTr
│   │   │   ├── amos_0507.nii.gz
│   │   │   ├── amos_0508.nii.gz
│   │   │   ├── ...
│   │   ├── dataset.json
│   ├── ...
"""
base = join(os.sep.join(__file__.split(os.sep)[:-3]), 'data') 
# or you can set your own path, e.g., base = '/home/user_name/Documents/segmentation_runtime/data'
seg_raw = join(base, 'seg_raw') # os.environ.get('seg_raw')
seg_preprocessed = join(base, 'seg_preprocessed') # os.environ.get('seg_preprocessed')
seg_results = join(base, 'seg_results') # os.environ.get('seg_results')

if seg_raw is None:
    print("seg_raw is not defined and nnU-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set "
          "this up properly.")

if seg_preprocessed is None:
    print("seg_preprocessed is not defined and nnU-Net can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how "
          "to set this up.")

if seg_results is None:
    print("seg_results is not defined and nnU-Net cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information "
          "on how to set this up.")
