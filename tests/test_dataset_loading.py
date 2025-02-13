import os
import time
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

# from diffusion_policy.dataset.umi_lazy_dataset import UmiLazyDataset
from diffusion_policy.dataset.umi_multi_dataset import UmiMultiDataset
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(
    config_path="../diffusion_policy/config/task/dataset",
    # config_name="umi_lazy_dataset",
    config_name="umi_multi_dataset"
)
def main(cfg: DictConfig):
    
    
    instantiated_cfg = hydra.utils.instantiate(cfg)
    # train_dataset: UmiLazyDataset = instantiated_cfg
    train_dataset: UmiMultiDataset = instantiated_cfg
    val_dataset = train_dataset.split_unused_episodes(
        remaining_ratio=1.0, other_used_episode_indices=[]
    )
    print(
        f"train_dataset length: {len(train_dataset)}, val_dataset length: {len(val_dataset)}"
    )
    # train_dataset.fit_normalizer()

    # assert train_dataset.normalizer is not None
    # print(train_dataset.normalizer.state_dict())

    # for i in range(1):
    #     train_data_shapes = {
    #         k: {k_: v_.shape for k_, v_ in v.items()}
    #         for k, v in train_dataset[i].items()
    #     }
    #     print(f"train_data_shapes: {train_data_shapes}")
    import numpy as np
    np.set_printoptions(precision=3, suppress=True)
    # for i in range(10):
    #     # print(train_dataset.normalizer.unnormalize(train_dataset[i])["output"]["action_0_tcp_xyz_wxyz"].numpy())
    #     print(train_dataset[i]["obs"]["camera0_rgb"].shape)

    obs_data = train_dataset[0]["obs"]
    for key, value in obs_data.items():
        print(key, value.shape)
    print("action", train_dataset[0]["action"].shape)

    


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
