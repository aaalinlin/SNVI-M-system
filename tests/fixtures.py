"""Synthetic configuration; contains no private dataset measurements."""


def synthetic_dataset():
    return {'channel_names': {'0': 'MRI'}, 'labels': {'background': 0, 'foreground': 1},
            'numTraining': 5, 'file_ending': '.nii.gz'}


def synthetic_plans():
    config = {
        'data_identifier': 'nnUNetPlans_3d_fullres', 'preprocessor_name': 'DefaultPreprocessor',
        'batch_size': 2, 'patch_size': [16, 16, 16], 'spacing': [1., 1., 1.],
        'normalization_schemes': ['ZScoreNormalization'], 'use_mask_for_norm': [False],
        'UNet_class_name': 'PlainConvUNet', 'UNet_base_num_features': 4,
        'n_conv_per_stage_encoder': [2, 2, 2], 'n_conv_per_stage_decoder': [2, 2],
        'num_pool_per_axis': [2, 2, 2], 'pool_op_kernel_sizes': [[1]*3, [2]*3, [2]*3],
        'conv_kernel_sizes': [[3]*3]*3, 'unet_max_num_features': 16, 'batch_dice': True,
        'resampling_fn_data': 'resample_data_or_seg_to_shape',
        'resampling_fn_seg': 'resample_data_or_seg_to_shape',
        'resampling_fn_probabilities': 'resample_data_or_seg_to_shape',
        'resampling_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0, 'force_separate_z': None},
        'resampling_fn_seg_kwargs': {'is_seg': True, 'order': 1, 'order_z': 0, 'force_separate_z': None},
        'resampling_fn_probabilities_kwargs': {'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': None},
    }
    return {'dataset_name': 'DatasetSynthetic', 'plans_name': 'nnUNetPlans',
            'image_reader_writer': 'SimpleITKIO', 'transpose_forward': [0, 1, 2],
            'transpose_backward': [0, 1, 2], 'label_manager': 'LabelManager',
            'foreground_intensity_properties_per_channel': {'0': {}},
            'configurations': {'3d_fullres': config}}
