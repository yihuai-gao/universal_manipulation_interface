from typing import Dict, Optional
import torch
import numpy as np
import copy
import hydra
import sys
import os

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
            dataset_path, keys=['robot_0_camera_images', 'robot_0_tcp_xyz_wxyz', 'robot_0_gripper_width', 'action_0_tcp_xyz_wxyz', 'action_0_gripper_width'])


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

            if key.endswith('eef_pos'):
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
        data = {
            'action': np.concatenate([self.replay_buffer['action_0_tcp_xyz_wxyz'], self.replay_buffer['action_0_gripper_width']], axis=-1),
            'proprioception': np.concatenate([self.replay_buffer['robot_0_tcp_xyz_wxyz'], self.replay_buffer['robot_0_gripper_width']], axis=-1),
            'image': self.replay_buffer['robot_0_camera_images']
        }

        dim_a = data['action'].shape[-1]
        action_normalizers = list()
        action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data['action'][..., i * dim_a: i * dim_a + 3])))              # pos
        action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data['action'][..., i * dim_a + 3: (i + 1) * dim_a - 1]))) # rot
        action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data['action'][..., (i + 1) * dim_a - 1: (i + 1) * dim_a])))  # gripper

        normalizer['image'] = get_image_identity_normalizer()
        normalizer['action'] = concatenate_normalizer(action_normalizers)

        normalizer['proprioception'] = concatenate_normalizer(action_normalizers)


        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # proprioception = sample['state'][:,:2].astype(np.float32) # (proprioceptionx2, block_posex3)
        pose = np.concatenate([sample['robot_0_tcp_xyz_wxyz'], sample['robot_0_gripper_width']], axis=-1).astype(np.float32)
        agent_action = sample['action']
        # image = np.moveaxis(sample['img'],-1,1)/255
        image = np.moveaxis(sample['robot_0_camera_images'].astype(np.float32).squeeze(1),-1,1)/255

        data = {
            'obs': {
                'image': image, # T, 3, 224, 224
                'proprioception': pose, # T, 8 (x,y,z,qx,qy,qz,qw,gripper_width)
            },
            'action': agent_action # T, 8 (x,y,z,qx,qy,qz,qw,gripper_width)
        }
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
    dataset_path = os.path.expanduser('/home/yihuai/robotics/repositories/mujoco/mujoco-env/data/collect_heuristic_data/2024-12-25_10-08-27_100episodes/merged_data.zarr')
    print(cfg.task.shape_meta)
    dataset = MujocoImageDataset(cfg.task.shape_meta, dataset_path)
    
    print(dataset[0])
    for key, value in dataset[0]["obs"].items():
        print(key, value.shape)

    print(dataset[0]['action'].shape)

if __name__ == '__main__':


    main()