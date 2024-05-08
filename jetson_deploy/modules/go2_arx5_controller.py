import enum
from typing import Optional
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

import time

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    RESET_TO_HOME = 3
    ADD_WAYPOINT = 4
    UPDATE_TRAJECTORY = 5


class Go2Arx5Listener(Node):
    def __init__(self):
        super().__init__('go2_arx5_state_listener')
        # qos_profile = QoSProfile(depth=1, history=rclpy.qos.HistoryPolicy.KEEP_LAST)
        self.eef_state_sub = self.create_subscription(
            String,  # Replace with your message type
            'eef_state',  # Replace with your topic name
            self.listener_callback,
            10)
        
        
        self.received_msg = None
    
    def listener_callback(self, msg):
        self.received_msg = msg

    def get_dict(self):
        assert self.received_msg is not None
        return { # TODO: check the actual message type
            "ActualTCPPose": self.received_msg.actual_pose,
            "gripper_position": self.received_msg.gripper_position,
            "robot_receive_timestamp": self.received_msg.receive_timestamp,
            "robot_timestamp": self.received_msg.timestamp
        }


class Go2Arx5Publisher(Node):

    def __init__(self):
        super().__init__('go2_arx5_traj_publisher')

        self.eef_traj_pub = self.create_publisher(
            String,  # Replace with your message type
            'eef_traj',  # Replace with your topic name
            100)
    def publish_target_traj(self, pose_traj, gripper_traj, timestamps):
        ...



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

        self.pub_node = Go2Arx5Publisher()
        rclpy.init()

        # Should be initialized in the subprocess
        self.sub_node: Go2Arx5Listener


    
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[Go2Arx5Controller] Controller process spawned at {self.pid}")

    
    def stop(self, wait=True):
        if wait:
            self.stop_wait()
        rclpy.shutdown()
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
        self.pub_node.publish_target_traj(pose_traj, gripper_traj, timestamps)


    # ========= main loop (only for listener update) ============
    def run(self):
        rclpy.init()
        self.sub_node = Go2Arx5Listener()
        iter_idx = 0
        try:
            while rclpy.ok():
                rclpy.spin_once(self.sub_node)
                if self.sub_node.received_msg is not None:
                    # update ring buffer
                    self.ring_buffer.put(self.sub_node.get_dict())
                    self.sub_node.received_msg = None
                    if iter_idx == 0:
                        self.ready_event.set()
                    iter_idx += 1
        finally:
            rclpy.shutdown()
