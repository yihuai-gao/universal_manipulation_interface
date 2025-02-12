import os
import sys
from typing import Any, Union, cast
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
import zarr
import numpy as np
import torch
import numpy.typing as npt

from diffusion_policy.dataset.base_lazy_dataset import BaseLazyDataset, batch_type
from diffusion_policy.dataset.umi_lazy_dataset import UmiLazyDataset


class UmiMultiDataset(Dataset[batch_type]):
    """
    Multi-dataset data loader for the official UMI dataset.
    Example structure:

    dataset_0.zarr
    ├── data
    │   ├── camera0_rgb (N, 224, 224, 3) uint8
    │   ├── robot0_demo_end_pose (N, 6) float64
    │   ├── robot0_demo_start_pose (N, 6) float64
    │   ├── robot0_eef_pos (N, 3) float32
    │   ├── robot0_eef_rot_axis_angle (N, 3) float32
    │   └── robot0_gripper_width (N, 1) float32
    └── meta
        └── episode_ends (5,) int64
    dataset_1.zarr
    ├── data
    └── meta
    dataset_2.zarr
    ├── data
    └── meta
    """

    def __init__(self, dataset_root_dir: str, dataset_config: Union[dict[str, dict[str, Any]], DictConfig], **kwargs):

        if isinstance(dataset_config, DictConfig):
            dataset_config = cast(dict[str, dict[str, Any]], OmegaConf.to_container(dataset_config))
        self.dataset_config: dict[str, dict[str, Any]] = dataset_config
        self.dataset_root_dir: str = dataset_root_dir
        self.kwargs = kwargs

        self.datasets: list[UmiLazyDataset] = []
        for dataset_name, dataset_config in self.dataset_config.items():
            config = self.
            dataset = UmiLazyDataset(**dataset_config)
            self.datasets.append(dataset)
