from typing import Dict, Optional
import torch
import numpy as np
import copy
import hydra
import sys
import os

import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat, get_image_identity_normalizer, get_image_range_normalizer, get_range_normalizer_from_stat

from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask)

class MujocoImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            cache_dir: Optional[str]=None,
            pose_repr: dict={},
            action_padding: bool=False,
            temporally_independent_normalization: bool=False,
            repeat_frame_prob: float=0.0,
            seed: int=42,
            val_ratio: float=0.0,
            max_duration: Optional[float]=None
            ):
        
        self.replay_buffer = ReplayBuffer.copy_from_path(
            dataset_path, keys=['robot0_camera_images', 'robot0_tcp_xyz_wxyz', 'robot0_gripper_width', 'action0_tcp_xyz_wxyz', 'action0_gripper_width'])


        self.num_robot = 0
        rgb_keys = list()
        lowdim_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            # solve obs type
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

            if key.endswith('tcp_xyz_wxyz'):
                self.num_robot += 1

            # solve obs_horizon
            horizon = shape_meta['obs'][key]['horizon']
            key_horizon[key] = horizon

            # solve latency_steps
            latency_steps = shape_meta['obs'][key]['latency_steps']
            key_latency_steps[key] = latency_steps

            # solve down_sample_steps
            down_sample_steps = shape_meta['obs'][key]['down_sample_steps']
            key_down_sample_steps[key] = down_sample_steps

        # solve action
        key_horizon['action'] = shape_meta['action']['horizon']
        key_latency_steps['action'] = shape_meta['action']['latency_steps']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        
        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            if not 'wrt' in key:
                self.sampler_lowdim_keys.append(key)
    
        for key in self.replay_buffer.keys():
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                self.sampler_lowdim_keys.append(key)
                query_key = key.split('_')[0] + '_eef_pos'
                key_horizon[key] = shape_meta['obs'][query_key]['horizon']
                key_latency_steps[key] = shape_meta['obs'][query_key]['latency_steps']
                key_down_sample_steps[key] = shape_meta['obs'][query_key]['down_sample_steps']


        self.sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=repeat_frame_prob,
            max_duration=max_duration
            )
        

        self.shape_meta = shape_meta
        self.replay_buffer = self.replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False




    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            repeat_frame_prob=self.repeat_frame_prob,
            max_duration=self.max_duration
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()

        # enumerate the dataset and save low_dim data
        data_cache = {key: list() for key in self.lowdim_keys + ['action']}
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=32,
        )
        for batch in tqdm.tqdm(dataloader, desc='iterating dataset to get normalization'):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch['obs'][key]))
            data_cache['action'].append(copy.deepcopy(batch['action']))
        self.sampler.ignore_rgb(False)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B*T, D)

        # action
        assert data_cache['action'].shape[-1] % self.num_robot == 0
        dim_a = data_cache['action'].shape[-1] // self.num_robot
        action_normalizers = list()
        for i in range(self.num_robot):
            action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., i * dim_a: i * dim_a + 3])))              # pos
            action_normalizers.append(get_identity_normalizer_from_stat(array_to_stats(data_cache['action'][..., i * dim_a + 3: (i + 1) * dim_a - 1]))) # rot
            action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., (i + 1) * dim_a - 1: (i + 1) * dim_a])))  # gripper

        normalizer['action'] = concatenate_normalizer(action_normalizers)

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])

            if key.endswith('pos') or 'pos_wrt' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('pos_abs'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('rot_axis_angle') or 'rot_axis_angle_wrt' in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('gripper_width'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('xyz_wxyz'):
                pos_normalizer = get_range_normalizer_from_stat(array_to_stats(data_cache[key][..., :3]))
                rot_normalizer = get_identity_normalizer_from_stat(array_to_stats(data_cache[key][..., 3:]))
                this_normalizer = concatenate_normalizer([pos_normalizer, rot_normalizer])
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # proprioception = sample['state'][:,:2].astype(np.float32) # (proprioceptionx2, block_posex3)
        # image = np.moveaxis(sample['img'],-1,1)/255

        data = {
            'obs': {
                'robot0_tcp_xyz_wxyz': sample['robot0_tcp_xyz_wxyz'].astype(np.float32), # T, 7 (x,y,z,qx,qy,qz,qw)
                'robot0_gripper_width': sample['robot0_gripper_width'].astype(np.float32), # T, 1
            },
            'action': sample['action'].astype(np.float32) # T, 8 (x,y,z,qx,qy,qz,qw,gripper_width)
        }
        if 'robot0_camera_images' in sample:
            image = np.moveaxis(sample['robot0_camera_images'].astype(np.float32).squeeze(1),-1,1)/255
            data['obs']['robot0_camera_images'] = image # T, 3, 224, 224
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    config_path="../config", config_name="train_diffusion_unet_timm_mujoco_workspace.yaml", version_base=None
)
def main(cfg):
    import os
    dataset_path = os.path.expanduser('/home/yihuai/robotics/repositories/mujoco/mujoco-env/data/collect_heuristic_data/2024-12-26_16-56-50_100episodes/merged_data.zarr')
    print(cfg.task.shape_meta)
    dataset = MujocoImageDataset(cfg.task.shape_meta, dataset_path)
    
    # print(dataset[0])
    for key, value in dataset[0]["obs"].items():
        print(key, value.shape)

    print(dataset[0]['action'].shape)

if __name__ == '__main__':


    main()