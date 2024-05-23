import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)

import enum
from typing import Optional
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from robot_state.msg import EEFState, EEFTraj
import time
from transforms3d import quaternions
import copy
import logging
class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    RESET_TO_HOME = 3
    ADD_WAYPOINT = 4
    UPDATE_TRAJECTORY = 5


class Go2Arx5Listener(Node):
    def __init__(self, shared_start_time: mp.Value):
        super().__init__('go2_arx5_state_listener')
        # qos_profile = QoSProfile(depth=1, history=rclpy.qos.HistoryPolicy.KEEP_LAST)
        
        self.eef_state_sub = self.create_subscription(
            EEFState, "go2_arx5/eef_state", self.listener_callback, 10
        )

        self.shared_start_time = shared_start_time
        self.received_msg = None
        self.msg_dict = None
        self.shared_start_time_update_cnt = 0
    
    def listener_callback(self, msg:EEFState):
        self.received_msg = msg

        # Update start time
        new_est_start_time = time.time() - float(msg.tick)/1000
        self.shared_start_time_update_cnt += 1
        # apply delta update to shared_start_time
        if self.shared_start_time_update_cnt > 1 and abs(new_est_start_time - self.shared_start_time.value) > 0.1:
            logging.warning(f"Shared start time is inconsistent: {new_est_start_time=} {self.shared_start_time.value=} {self.shared_start_time_update_cnt=}")
        else:
            self.shared_start_time.value = self.shared_start_time.value + (new_est_start_time - self.shared_start_time.value) / self.shared_start_time_update_cnt

        eef_translation = msg.eef_pose[:3]
        eef_rotation_quat = msg.eef_pose[3:]
        vec, theta = quaternions.quat2axangle(eef_rotation_quat)
        eef_rotation_vec = vec * theta
        actual_tcp_pose = np.zeros(6)
        actual_tcp_pose[:3] = eef_translation
        actual_tcp_pose[3:] = eef_rotation_vec
        self.msg_dict = { # TODO: check the actual message type
            "ActualTCPPose": actual_tcp_pose,
            "gripper_position": msg.gripper_pos,
            "robot_receive_timestamp": time.time(),
            "robot_timestamp": float(msg.tick)/1000 + self.shared_start_time.value
        }


    def get_dict(self):
        return copy.deepcopy(self.msg_dict)

        


class Go2Arx5Publisher(Node):

    def __init__(self):
        super().__init__('go2_arx5_traj_publisher')

        self.eef_traj_pub = self.create_publisher(
            EEFTraj,  # Replace with your message type
            'go2_arx5/eef_traj',  # Replace with your topic name
            10)
    def publish_target_traj(self, pose_traj: np.ndarray, gripper_traj: np.ndarray, robot_ticks_s: np.ndarray):
        # pose_traj: (N, 6), gripper_traj: (N,), robot_ticks_s: (N,)
        if len(pose_traj) == 0:
            return
        # print(f"{pose_traj.shape=} {gripper_traj.shape=} {robot_ticks_s.shape=}")
        assert isinstance(pose_traj, np.ndarray)
        assert isinstance(gripper_traj, np.ndarray)
        assert isinstance(robot_ticks_s, np.ndarray)
        assert len(pose_traj.shape) == 2
        assert pose_traj.shape[1] == 6
        assert gripper_traj.shape[0] == pose_traj.shape[0] == robot_ticks_s.shape[0]
        
        # Convert rotvec to quaternion
        eef_traj = EEFTraj()
        eef_frames = []
        for i in range(len(pose_traj)):
            pose = pose_traj[i]
            eef_frame = EEFState()
            theta = np.linalg.norm(pose[3:])
            if theta > 1e-6:
                vec = pose[3:] / theta
            else:
                vec = np.zeros(3)
            eef_frame.eef_pose[:3] = pose[:3]
            eef_frame.eef_pose[3:] = quaternions.axangle2quat(vec, theta)
            eef_frame.gripper_pos = float(gripper_traj[i])
            eef_frame.tick = int(robot_ticks_s[i] * 1000)
            eef_frames.append(eef_frame)
            
        eef_traj.traj = eef_frames
        self.eef_traj_pub.publish(eef_traj)



class Go2Arx5Controller(mp.Process):
    def __init__(
            self,
            shm_manager,
            launch_timeout: float = 10,
            frequency: float = 100,
            get_max_k: Optional[int] = None,
            verbose=True,
            receive_latency: float = 0.0,
        ):
        super().__init__(name="Go2Arx5Controller")
        
        self.shm_manager = shm_manager
        self.verbose = verbose

        # build ring buffer
        example = dict()
        example["ActualTCPPose"] = np.zeros(6)
        example['gripper_position'] = 0.0

        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()

        if get_max_k is None:
            get_max_k = int(frequency * 5)
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        self.launch_timeout = launch_timeout
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.receive_latency = receive_latency

        self.pub_node: Optional[Go2Arx5Publisher] = None

        # Should be initialized in the subprocess
        self.sub_node: Go2Arx5Listener


        # create a shared float start_time to convert from robot tick to time.time()
        self.robot_start_time = mp.Value('d', 0.0) # time.time() when the robot tick is 0

    
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        rclpy.init()
        self.pub_node = Go2Arx5Publisher()
        print("Pub node initialized")
        if self.verbose:
            print(f"[Go2Arx5Controller] Controller process spawned at {self.pid}")

    
    def stop(self, wait=True):
        rclpy.shutdown()
        if wait:
            self.stop_wait()
        if self.verbose:
            print(f"[Go2Arx5Controller] Controller process terminated at {self.pid}")

    def start_wait(self):
        print(f"[Go2Arx5Controller] Waiting for controller process to be ready")
        print(f"{self.launch_timeout=}")
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
     
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= command methods ============
    def send_trajectory(self, pose_traj, gripper_traj, timestamps):

        robot_ticks_s = timestamps - self.robot_start_time.value

        robot_ticks_s += 0.1
        if self.pub_node is not None:
            self.pub_node.publish_target_traj(pose_traj, gripper_traj, robot_ticks_s)
        


    # ========= main loop (only for listener update) ============
    def run(self):
        rclpy.init()
        self.sub_node = Go2Arx5Listener(self.robot_start_time)
        iter_idx = 0
        # try:
        while rclpy.ok():
            rclpy.spin_once(self.sub_node)
            if self.sub_node.received_msg is not None:
                # update ring buffer
                state_dict = self.sub_node.get_dict()
                if state_dict is not None:
                    # print(f"{state_dict['robot_timestamp']=}")
                    self.ring_buffer.put(state_dict)
                self.sub_node.received_msg = None
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
        # finally:
        #     rclpy.shutdown()


if __name__ == "__main__":
    # Test subscriber output
    with SharedMemoryManager() as shm_manager:
        with Go2Arx5Controller(shm_manager=shm_manager) as go2_arx5_controller:
            while True:
                state = go2_arx5_controller.get_state()
                tcp_pose = state["ActualTCPPose"]
                gripper_pos = state["gripper_position"]
                timestamp = state["robot_timestamp"]
                go2_arx5_controller.send_trajectory(
                    tcp_pose.reshape(1,6).repeat(1, axis=0),
                    gripper_pos.reshape(1,).repeat(1, axis=0),
                    timestamp.reshape(1,).repeat(1, axis=0),
                )
                time.sleep(0.05)


    