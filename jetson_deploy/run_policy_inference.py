import sys
import os
import time
import click
import numpy as np
import torch
import dill
import hydra
import zmq
from line_profiler import profile
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import get_real_obs_resolution, get_real_umi_action
from diffusion_policy.common.pytorch_util import dict_apply

@profile
@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
def main(input, output):
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    print(f"{cfg.name=}, {cfg._target_=}, {cfg.policy._target_=}")
    get_class_start_time = time.monotonic()
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    print(f"get_class: {time.monotonic() - get_class_start_time:.3f}s")
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=["traced_model"], include_keys=None)
    policy = workspace.model
    if cfg.training.use_ema:
        print("Using EMA model")
        policy = workspace.ema_model
    policy.num_inference_steps = 16 # DDIM inference iterations
    obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr
    print('obs_pose_rep', obs_pose_rep)
    print('action_pose_repr', action_pose_repr)
    device = torch.device('cuda')
    policy.eval().to(device)

    iter_idx = 0
    policy.reset()
    while True:
        try:
            obs_dict_np = np.load(f'{output}/obs/obs_dict_{iter_idx}.npy', allow_pickle=True).item()
        except:
            break
        with torch.no_grad():
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            result = policy.predict_action(obs_dict)
            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
            # action = get_real_umi_action(raw_action, obs_dict_np, action_pose_repr)
        iter_idx += 6
        # print(f'iter {iter_idx}: raw_action: {raw_action}')
                

if __name__ == '__main__':
    main()