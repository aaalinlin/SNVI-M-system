"""Keep inference parameters while excluding patient identifiers and training records."""
from copy import deepcopy


def inference_metadata(plans, dataset, configuration='3d_fullres'):
    from segcore.utilities.plans_handling.plans_handler import PlansManager
    cfg = deepcopy(PlansManager(plans).get_configuration(configuration).configuration)
    if cfg.get('previous_stage') is not None:
        raise ValueError('Cascaded models are not supported by this exporter')
    if cfg['normalization_schemes'] != ['ZScoreNormalization']:
        raise ValueError('Statistics-free export requires single-channel MRI ZScoreNormalization')
    keep = ['preprocessor_name', 'patch_size', 'spacing', 'normalization_schemes',
            'use_mask_for_norm', 'UNet_class_name', 'UNet_base_num_features',
            'n_conv_per_stage_encoder', 'n_conv_per_stage_decoder', 'num_pool_per_axis',
            'pool_op_kernel_sizes', 'conv_kernel_sizes', 'unet_max_num_features',
            'resampling_fn_data', 'resampling_fn_data_kwargs', 'resampling_fn_seg',
            'resampling_fn_seg_kwargs', 'resampling_fn_probabilities', 'resampling_fn_probabilities_kwargs']
    minimal = {key: deepcopy(plans[key]) for key in
               ['image_reader_writer', 'transpose_forward', 'transpose_backward']}
    minimal['label_manager'] = plans.get('label_manager', 'LabelManager')
    minimal['configurations'] = {configuration: {key: cfg[key] for key in keep}}
    # The reader indexes channel zero, but per-image Z-score ignores cohort statistics.
    minimal['foreground_intensity_properties_per_channel'] = {'0': {}}
    if (set(dataset['channel_names']) != {'0'} or len(dataset['labels']) != 2
            or sorted(dataset['labels'].values()) != [0, 1]):
        raise ValueError('Expected one MRI channel with binary labels')
    metadata = {'channel_names': {'0': 'MRI'}, 'labels': {'background': 0, 'foreground': 1},
                'file_ending': dataset['file_ending']}
    return minimal, metadata
