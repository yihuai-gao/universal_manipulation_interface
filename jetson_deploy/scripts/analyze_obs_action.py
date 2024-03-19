import numpy as np
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from umi.common.pose_util import mat_to_pose, pose10d_to_mat, pose_to_mat
import zmq
from umi.real_world.real_inference_util import get_real_umi_action
output_dir = "data_local/arx_20240320/vit_l_mirror_swap"
episode_id = 0

np.set_printoptions(precision=4, suppress=True)


# obs_data = {
#     "obs_dict_np": obs_dict_np, 
#     "obs_pose_rep": obs_pose_rep, 
#     "obs": obs, 
#     "episode_start_pose": episode_start_pose,
#     "tx_robot1_robot0": tx_robot1_robot0
# }

# action_data = {
#     "action": action,
#     "raw_action": raw_action,
#     "action_pose_repr": action_pose_repr
# }

step_num = 0
inference_step_num = 6

while True:
    try:
        action_data = np.load(f"{output_dir}/action/{episode_id}/{step_num}.npy", allow_pickle=True).item()
        obs_data = np.load(f"{output_dir}/obs/{episode_id}/{step_num}.npy", allow_pickle=True).item()
    except FileNotFoundError:
        break
    obs = obs_data["obs"]
    action = action_data["action"]
    raw_action = action_data["raw_action"]
    action_pose_repr = action_data["action_pose_repr"]
    # calc_action = get_real_umi_action(raw_action, obs, action_pose_repr)

    n_robots = int(raw_action.shape[-1] // 10)
    env_action = list()
    for robot_idx in range(n_robots):
        # convert pose to mat
        pose_mat = pose_to_mat(np.concatenate([
            obs[f'robot{robot_idx}_eef_pos'][-1],
            obs[f'robot{robot_idx}_eef_rot_axis_angle'][-1]
        ], axis=-1))
        # print(f"{obs[f'robot{robot_idx}_eef_pos'][-1]=}", f"{obs[f'robot{robot_idx}_eef_rot_axis_angle'][-1]=}")
        rpy = obs[f'robot{robot_idx}_eef_rot_axis_angle'][-1]

        start = robot_idx * 10
        action_pose10d = raw_action[..., start:start+9]
        action_grip = raw_action[..., start+9:start+10]
        action_pose_mat = pose10d_to_mat(action_pose10d)
        relative_action_pose = mat_to_pose(action_pose_mat)

        # solve relative raw_action
        action_mat = convert_pose_mat_rep(
            action_pose_mat, 
            base_pose_mat=pose_mat,
            pose_rep=action_pose_repr,
            backward=True)

        # convert raw_action to pose
        action_pose = mat_to_pose(action_mat)
        env_action.append(action_pose)
        env_action.append(action_grip)

    env_action = np.concatenate(env_action, axis=-1)
    # print(np.all(action == calc_action))
    print(f"Step {step_num}")
    # print(f"Obs: {obs}")
    print(f"")
    # print(f"Episode start pose: {obs_data['episode_start_pose']}")
    print(f"{obs[f'robot{0}_eef_pos']=}")
    # print(f"{obs[f'robot{0}_eef_rot_axis_angle']=}")
    # print(f"{relative_action_pose=}")
    print(f"Action: {action}")
    monotonic_step_range = 5
    monotonic_cnt = 0
    for k in range(6):
        if np.all(action[1:monotonic_step_range+1, k] >= action[:monotonic_step_range, k] - 0.0001):
            monotonic_cnt += 1
        elif np.all(action[1:monotonic_step_range+1, k] <= action[:monotonic_step_range, k] + 0.0001):
            monotonic_cnt += 1
    # print(f"{monotonic_cnt=}")


    step_num += inference_step_num

# ref_output_dir = "data_local/wild_cup_20240203/vit_l_mirror_swap"
# ref_obs_dict = np.load(f"{ref_output_dir}/obs/obs_dict_360.npy", allow_pickle=True).item()
# print(ref_obs_dict.keys())
# context = zmq.Context()
# socket = context.socket(zmq.REQ)
# socket.connect(f"tcp://iris-robot-ws-2:8766")
# socket.send_pyobj(ref_obs_dict)
# ref_raw_action = socket.recv_pyobj()

# action_pose10d = ref_raw_action[..., :9]
# action_grip = ref_raw_action[..., 9:10]
# action_pose_mat = pose10d_to_mat(action_pose10d)
# ref_relative_action_pose = mat_to_pose(action_pose_mat)
# print(f"{ref_relative_action_pose=}")
# ref_action = get_real_umi_action(ref_raw_action, ref_obs, "relative")
# print(f"{ref_action=}")