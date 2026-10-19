import _bootstrap
import argparse
import os

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Predict NIfTI cases with exported PolarityD models")
    p.add_argument("--model-folder", required=True)
    p.add_argument("--input", required=True, help="Folder containing CASE_0000.nii.gz")
    p.add_argument("--output", required=True)
    p.add_argument("--folds", type=int, nargs="+", default=[0])
    p.add_argument("--gpu", default="0")
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch
    from segcore.inference.predict_from_raw_data import SegPredictor
    torch.set_num_threads(2)
    predictor = SegPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                               perform_everything_on_device=True, device=torch.device("cuda"))
    predictor.initialize_from_trained_model_folder(a.model_folder, a.folds, "checkpoint_final.pth")
    predictor.predict_from_files(a.input, a.output, save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2)
