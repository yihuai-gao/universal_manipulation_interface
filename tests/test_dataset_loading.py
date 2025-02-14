import os
import time
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from diffusion_policy.dataset.umi_lazy_dataset import UmiLazyDataset
from diffusion_policy.dataset.umi_multi_dataset import UmiMultiDataset
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import cv2
OmegaConf.register_new_resolver("eval", eval)

@hydra.main(
    config_path="../diffusion_policy/config/task/dataset",
    # config_name="umi_lazy_dataset",
    config_name="umi_multi_dataset"
)
def main(cfg: DictConfig):
    
    
    instantiated_cfg = hydra.utils.instantiate(cfg)
    train_dataset: UmiLazyDataset = instantiated_cfg
    # train_dataset: UmiMultiDataset = instantiated_cfg
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

    # obs_data = train_dataset[0]["obs"]
    # for key, value in obs_data.items():
    #     print(key, value.shape)
    # print("action", train_dataset[0]["action"].shape)
    image_dir = "/scratch/m000073/yihuai/robotics/repositories/policies/imitation-learning-policies/prior_works/universal_manipulation_interface/data/test/"
    os.makedirs(image_dir, exist_ok=True)
    # for i in range(10000):
    #     # print(train_dataset[i]["obs"]["camera0_rgb"].shape)
        
        
    #     camera0_rgb = train_dataset[i]["obs"]["camera0_rgb"]
    #     concat_img = torch.cat([camera0_rgb[0], camera0_rgb[1]], dim=2)
    #     img = concat_img.numpy().transpose(1, 2, 0)*255.0
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8)
    #     print(img.shape)
    #     cv2.imwrite(f"{image_dir}/test_dataset_loading_{i}.png", img)
    #     cv2.waitKey(0)
    #     time.sleep(1)

    dataloader = train_dataset.get_dataloader()
    transforms = train_dataset.transforms
    for batch in dataloader:
        batch = transforms.apply(batch)
        camera0_rgb = batch["obs"]["camera0_rgb"]
        for i in range(camera0_rgb.shape[0]):
            concat_img = torch.cat([camera0_rgb[i][0], camera0_rgb[i][1]], dim=2)
            img = concat_img.numpy().transpose(1, 2, 0)*255.0
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8)
            cv2.imwrite(f"{image_dir}/test_dataset_loading_{i}.png", img)
            cv2.waitKey(0)
            time.sleep(0.1)
            print(batch["obs"]["camera0_rgb"].shape)
        break


    


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
