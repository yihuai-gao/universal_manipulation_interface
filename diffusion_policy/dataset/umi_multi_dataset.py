import json
import os
from typing import Any, Dict, Optional, Union, cast
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from diffusion_policy.dataset.base_lazy_dataset import BaseLazyDataset, batch_type
from diffusion_policy.dataset.umi_lazy_dataset import UmiLazyDataset
from copy import deepcopy


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

    def __init__(
        self,
        dataset_root_dir: str,
        used_episode_indices_file: str,
        dataset_configs: Union[dict[str, dict[str, Any]], DictConfig],
        **base_config: Union[dict[str, Any], DictConfig],
    ):

        self.dataset_root_dir: str = dataset_root_dir

        if isinstance(dataset_configs, DictConfig):
            dataset_configs = cast(
                dict[str, dict[str, Any]], OmegaConf.to_container(dataset_configs)
            )
        self.dataset_configs: dict[str, dict[str, Any]] = dataset_configs

        if used_episode_indices_file != "":
            assert used_episode_indices_file.endswith(
                ".json"
            ), "used_episode_indices_file must be a json file"
            with open(used_episode_indices_file, "r") as f:
                used_episode_indices_dict: dict[str, list[int]] = json.load(f)
            for name, config in self.dataset_configs.items():
                config["include_episode_indices"] = used_episode_indices_dict[name]
                if "include_episode_num" in config:
                    assert (
                        len(config["include_episode_indices"])
                        == config["include_episode_num"]
                    ), f"include_episode_num {config['include_episode_num']} does not match the length of include_episode_indices {len(config['include_episode_indices'])} for dataset {name}"

        if isinstance(base_config, DictConfig):
            base_config = cast(dict[str, Any], OmegaConf.to_container(base_config))
        self.base_config: dict[str, Any] = base_config

        self.datasets: list[UmiLazyDataset] = []
        for dataset_name, dataset_config in self.dataset_configs.items():
            print(f"Initializing dataset: {dataset_name}")
            config = deepcopy(self.base_config)
            config.update(deepcopy(dataset_config))
            config["zarr_path"] = os.path.join(
                self.dataset_root_dir, dataset_name + ".zarr"
            )
            config["name"] = dataset_name
            dataset = UmiLazyDataset(**config)
            self.datasets.append(dataset)

        self.index_pool: list[tuple[int, int]] = []
        """
        First value: dataset index
        Second value: data index in the corresponding dataset
        """
        self._create_index_pool()

    def _create_index_pool(self):
        self.index_pool = []
        for dataset_idx, dataset in enumerate(self.datasets):
            self.index_pool.extend((dataset_idx, i) for i in range(len(dataset)))

    def __len__(self):
        return len(self.index_pool)

    def __getitem__(self, idx: int) -> batch_type:
        dataset_idx, data_idx = self.index_pool[idx]
        return self.datasets[dataset_idx][data_idx]

    def split_unused_episodes(
        self,
        remaining_ratio: float = 1.0,
        other_used_episode_indices: Optional[list[int]] = None,
    ):
        unused_dataset = deepcopy(self)
        unused_dataset.index_pool = []
        unused_dataset.datasets = []
        for dataset_idx, dataset in enumerate(self.datasets):
            unused_dataset.datasets.append(
                dataset.split_unused_episodes(
                    remaining_ratio, other_used_episode_indices
                )
            )
        unused_dataset._create_index_pool()

        return unused_dataset

    def get_dataloader(self):
        return DataLoader(self, self.base_config["dataloader_cfg"])
