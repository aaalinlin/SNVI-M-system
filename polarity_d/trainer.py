"""A common seeded protocol for baseline, fixed routing and adaptive D reading."""
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from segcore.training.trainer.base_trainer import SegTrainer as nnUNetTrainer
from polarity_d.model import PolarityDUNet3D
from polarity_d.runtime import atomic_json, status


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ProtocolTrainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device("cuda")):
        if fold == "all" or int(fold) not in range(5):
            raise ValueError("Use a held-out fold index 0..4; 'all' is not cross-validation")
        self.base_seed = int(os.environ.get("PD_SEED", "20261017")) + 10000 * int(fold)
        seed_all(self.base_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # PyTorch 2.4 CUDA CrossEntropy lacks a deterministic implementation.
        # Preserve the original loss; warn and record this reproducibility limit.
        torch.use_deterministic_algorithms(True, warn_only=True)
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        if self.device.type == "cuda":
            self.grad_scaler = torch.amp.GradScaler("cuda", init_scale=1024)
        self.num_epochs = int(os.environ.get("PD_EPOCHS", "700"))
        self.initial_lr = 0.01
        self.weight_decay = 3e-5
        self.save_every = 10
        self.num_iterations_per_epoch = 700
        self.num_val_iterations_per_epoch = 50
        self.da_workers = int(os.environ.get("PD_DA_WORKERS", "4"))
        if self.da_workers < 1:
            raise ValueError("PD_DA_WORKERS must be positive")

    def _do_i_compile(self):
        return False

    def get_dataloaders(self):
        scales = self._get_deep_supervision_scales()
        rotation, dummy, initial_patch, mirrors = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        common = dict(is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
                      regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
                      ignore_label=self.label_manager.ignore_label)
        train_transform = self.get_training_transforms(
            self.configuration_manager.patch_size, rotation, scales, mirrors, dummy,
            order_resampling_data=3, order_resampling_seg=1,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm, **common)
        val_transform = self.get_validation_transforms(scales, **common)
        dl_train, dl_val = self.get_plain_dataloaders(initial_patch, len(self.configuration_manager.patch_size))

        # NonDetMultiThreadedAugmenter. Workers are reseeded each epoch below.
        return (MultiThreadedAugmenter(dl_train, train_transform, self.da_workers, 2,
                                      seeds=[self.base_seed + i for i in range(self.da_workers)],
                                      pin_memory=self.device.type == "cuda"),
                MultiThreadedAugmenter(dl_val, val_transform, 1, 2,
                                      seeds=[self.base_seed + 500000],
                                      pin_memory=self.device.type == "cuda"))

    def _restart_generator(self, generator, stream):
        generator._finish()
        generator.seeds = [self.base_seed + 1000000 * stream + 1000 * self.current_epoch + i
                           for i in range(generator.num_processes)]

    def on_train_epoch_start(self):
        seed_all(self.base_seed + 1000 * self.current_epoch)
        self._restart_generator(self.dataloader_train, 0)
        super().on_train_epoch_start()

    def on_validation_epoch_start(self):
        # Release queued training patches while the validation workers run.
        self.dataloader_train._finish()
        self._restart_generator(self.dataloader_val, 1)
        super().on_validation_epoch_start()

    def on_epoch_end(self):
        self.dataloader_val._finish()
        super().on_epoch_end()
        logs = self.logger.my_fantastic_logging
        status(Path(self.output_folder) / "status.json", "training", fold=int(self.fold),
               completed_epochs=self.current_epoch, max_epochs=self.num_epochs,
               train_loss=float(logs["train_losses"][-1]), val_loss=float(logs["val_losses"][-1]),
               patch_dice=[float(x) for x in logs["dice_per_class_or_region"][-1]],
               epoch_seconds=float(logs["epoch_end_timestamps"][-1] - logs["epoch_start_timestamps"][-1]))

    def save_checkpoint(self, filename):
        if self.disable_checkpointing:
            return
        if self.is_ddp:
            raise NotImplementedError("Reproducible launcher uses one GPU per fold")
        checkpoint = {
            "network_weights": self.network.state_dict(), "optimizer_state": self.optimizer.state_dict(),
            "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            "logging": self.logger.get_checkpoint(), "_best_ema": self._best_ema,
            "current_epoch": self.current_epoch + 1, "init_args": self.my_init_kwargs,
            "trainer_name": self.__class__.__name__,
            "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
            "polarity_d_protocol": {"seed": self.base_seed, "epochs": self.num_epochs,
                                    "da_workers": self.da_workers, "epoch_reseed": True},
            "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                          "torch": torch.get_rng_state(),
                          "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
        temporary = str(filename) + ".tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, filename)

    def load_checkpoint(self, filename_or_checkpoint):
        checkpoint = (torch.load(filename_or_checkpoint, map_location="cpu", weights_only=False)
                      if isinstance(filename_or_checkpoint, str) else filename_or_checkpoint)
        expected = {"seed": self.base_seed, "epochs": self.num_epochs,
                    "da_workers": self.da_workers, "epoch_reseed": True}
        if checkpoint.get("polarity_d_protocol") != expected:
            raise ValueError("Resume requires the same seed, epoch budget and augmentation worker count")
        if not self.was_initialized:
            self.initialize()
        self.network.load_state_dict(checkpoint["network_weights"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None and checkpoint["grad_scaler_state"] is not None:
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])
        self.my_init_kwargs = checkpoint["init_args"]
        self.current_epoch = checkpoint["current_epoch"]
        self.logger.load_checkpoint(checkpoint["logging"])
        self._best_ema = checkpoint["_best_ema"]
        self.inference_allowed_mirroring_axes = checkpoint["inference_allowed_mirroring_axes"]
        rng = checkpoint["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])


class nnUNetTrainerPolarityDBaseline(ProtocolTrainer):



class nnUNetTrainerPolarityDFixed(ProtocolTrainer):
    @staticmethod
    def build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                   num_input_channels, enable_deep_supervision=True):
        model = nnUNetTrainer.build_network_architecture(
            plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision)
        return PolarityDUNet3D(model, adaptive=False)


class nnUNetTrainerPolarityDAdaptive(ProtocolTrainer):
    @staticmethod
    def build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                   num_input_channels, enable_deep_supervision=True):
        model = nnUNetTrainer.build_network_architecture(
            plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision)
        return PolarityDUNet3D(model, adaptive=True)
