import multiprocessing as mp
import enum
from multiprocessing.managers import SharedMemoryManager
from typing import Optional, cast
import numpy as np
from regex import F
from umi.common.precise_sleep import precise_wait
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
import time
import zmq
from jetson_deploy.modules.arx5_zmq_client import Arx5Client
import numpy.typing as npt
class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


class Arx5Controller(mp.Process):
    pass

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        robot_ip: str,
        robot_port: int,
        launch_timeout: float = 3,
        frequency: float = 200,
        get_max_k: Optional[int] = None,
        verbose: bool = False,
        receive_latency: float = 0.0,
    ):
        super().__init__(name="Arx5Controller")
        self.robot_ip = robot_ip
        self.robot_port = robot_port



        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'gripper_pos': 0.0,
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )
        self.verbose = verbose

        # build ring buffer
        receive_keys = [
            ('ActualTCPPose', 'ee_pose'),
            ('ActualQ', 'joint_pos'),
            ('ActualQd','joint_vel'),
            ('gripper_position', 'gripper_pos'),
        ]
        example = dict()
        for key, func_name in receive_keys:
            if 'joint' in func_name:
                example[key] = np.zeros(6)
            elif 'ee_pose' in func_name:
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
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
        self.frequency = frequency
        self.receive_latency = receive_latency

        # Will be initialized in the subprocess
        self.robot_client: Arx5Client


    

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[Arx5Controller] Controller process spawned at {self.pid}")

    
    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
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
    def servoL(self, pose:npt.NDArray[np.float64], gripper_pos: float, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'gripper_pos': gripper_pos,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose:npt.NDArray[np.float64], gripper_pos: float, target_time: float):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'gripper_pos': gripper_pos,
            'target_time': target_time
        }
        self.input_queue.put(message)

    

    # ========= main loop in process ============
    def run(self):
        self.robot_client = Arx5Client(self.robot_ip, self.robot_port)
        self.robot_client.reset_to_home()
        time.sleep(1)

        try:
            if self.verbose:
                print("[Arx5Controller] Controller process is ready")
            
            dt = 1/self.frequency
            self.robot_client.get_state()
            curr_pose = self.robot_client.ee_pose
            curr_gripper_pos = self.robot_client.gripper_pos
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=np.array([curr_t]),
                poses=np.array([curr_pose])
            )
            gripper_pos_interp = PoseTrajectoryInterpolator(
                times=np.array([curr_t]),
                poses=np.array([[curr_gripper_pos,0,0,0,0,0]])
            )


            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                t_now = time.monotonic()
                pose_cmd = pose_interp(t_now)
                gripper_cmd = float(gripper_pos_interp(t_now)[0])
                self.robot_client.set_ee_pose(pose_cmd, gripper_cmd)
                # print(f'arx5 controller sending pose: {pose_cmd} gripper: {gripper_cmd}')
                # self.robot_client.get_state()

                state = dict()
                for key, func_name in self.receive_keys:
                    state[key] = getattr(self.robot_client, func_name)
                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                try:
                    # process at most 1 command per cycle to maintain frequency
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    commands = {}
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        gripper_pos_interp = gripper_pos_interp.drive_to_waypoint(
                            pose=[command['gripper_pos'], 0, 0, 0, 0, 0],
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print(f"[Arx5Controller] New pose target: {target_pose} duration {duration:.3f}s")
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        target_time = time.monotonic() - time.time() + target_time
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=t_now,
                            last_waypoint_time=last_waypoint_time
                        )
                        gripper_pos_interp = gripper_pos_interp.schedule_waypoint(
                            pose=[command['gripper_pos'], 0, 0, 0, 0, 0],
                            time=target_time,
                            curr_time=t_now,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                        if self.verbose:
                            print(f"[Arx5Controller] New pose target: {target_pose} scheduled at {target_time:.3f}s")
                    else:
                        keep_running = False
                        print(f"[Arx5Controller] Unknown command {cmd}")
                        break
                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                if self.verbose:
                    print(f"[Arx5Controller] Actual frequency {1/(time.monotonic() - t_now)} Hz")
    

        finally:
            print("[Arx5Controller] Setting robot to damping")
            self.robot_client.set_to_damping()
            del self.robot_client
            self.ready_event.set()
            if self.verbose:
                print("[Arx5Controller] Controller process terminated")
