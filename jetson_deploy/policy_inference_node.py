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

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import get_real_obs_resolution, get_real_umi_action
from diffusion_policy.common.pytorch_util import dict_apply

class PolicyInferenceNode:
    def __init__(self, ckpt_path: str, ip: str, port: int):
        self.ckpt_path = ckpt_path
        if not self.ckpt_path.endswith('.ckpt'):
            self.ckpt_path = os.path.join(self.ckpt_path, 'checkpoints', 'latest.ckpt')
        payload = torch.load(open(self.ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        self.cfg = payload['cfg']
        print(f"Loading configure: {self.cfg.name}, workspace: {self.cfg._target_}, policy: {self.cfg.policy._target_}, model_name: {self.cfg.policy.obs_encoder.model_name}")
        self.obs_res = get_real_obs_resolution(self.cfg.task.shape_meta)
        self.get_class_start_time = time.monotonic()
        cls = hydra.utils.get_class(self.cfg._target_)
        self.workspace = cls(self.cfg)
        self.workspace: BaseWorkspace
        self.workspace.load_payload(payload, exclude_keys=["traced_model"], include_keys=None)
        self.policy:BaseImagePolicy = self.workspace.model
        if self.cfg.training.use_ema:
            self.policy = self.workspace.ema_model
            print("Using EMA model")
        self.policy.num_inference_steps = 16
        obs_pose_rep = self.cfg.task.pose_repr.obs_pose_repr
        action_pose_repr = self.cfg.task.pose_repr.action_pose_repr
        print('obs_pose_rep', obs_pose_rep)
        print('action_pose_repr', action_pose_repr)
        self.device = torch.device('cuda')
        self.policy.eval().to(self.device)
        self.policy.reset()
        self.ip = ip
        self.port = port

    def predict_action(self, obs_dict_np: dict):
        with torch.no_grad():
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
            result = self.policy.predict_action(obs_dict)
            action = result['action_pred'][0].detach().to('cpu').numpy()
        return action
    
    def run_node(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{self.ip}:{self.port}")
        print(f"PolicyInferenceNode is listening on {self.ip}:{self.port}")
        while True:
            obs_dict_np = socket.recv_pyobj()
            try:
                start_time = time.monotonic()
                action = self.predict_action(obs_dict_np)
                print(f'Inference time: {time.monotonic() - start_time:.3f} s')
            except Exception as e:
                print(f'Error: {e}')
                action = None
            send_start_time = time.monotonic()
            socket.send(action)
            print(f'Send time: {time.monotonic() - send_start_time:.3f} s')
    
@profile
@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
# @click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--ip', default="localhost")
@click.option('--port', default=8765, help="Port to listen on")
def main(input, ip, port):
    node = PolicyInferenceNode(input, ip, port)
    node.run_node()
                

if __name__ == '__main__':
    main()