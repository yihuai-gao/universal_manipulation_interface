from typing import Optional
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

import time

class Go2Arx5Node(Node):
    def __init__(self):
        super().__init__('one_message_subscriber')
        qos_profile = QoSProfile(depth=1, history=rclpy.qos.HistoryPolicy.KEEP_LAST)
        self.eef_state_sub = self.create_subscription(
            String,  # Replace with your message type
            'eef_state',  # Replace with your topic name
            self.listener_callback,
            qos_profile)
        
        self.eef_traj_pub = self.create_publisher(
            String,  # Replace with your message type
            'eef_traj',  # Replace with your topic name
            qos_profile)
        
        self.received_msg = None
    
    def listener_callback(self, msg):
        self.received_msg = msg



class Go2Arx5Controller:
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
        # self.node = Go2Arx5Node()

        # build ring buffer
        receive_keys = [
            ('ActualTCPPose', 'tcp_pose'),
            ('ActualQ', 'joint_pos'),
            ('ActualQd','joint_vel'),
            ('gripper_position', 'gripper_pos'),
        ]
        example = dict()
        for key, func_name in receive_keys:
            if 'joint' in func_name:
                example[key] = np.zeros(6)
            elif 'tcp_pose' in func_name:
                example[key] = np.zeros(6)
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

        rclpy.init()

    # ========= receive APIs =============
    def get_state(self):

    
    
    