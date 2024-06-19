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
from umi.shared_memory.shared_memory_queue import SharedMemoryQueue
from unitree_go.msg import WirelessController
import time
from transforms3d import quaternions
import copy
import logging
import traceback


KEY_ORDER = {
    0:"R1",
    1:"L1",
    2:"start",
    3:"select",
    4:"R2",
    5:"L2",
    8:"A",
    9:"B",
    10:"X",
    11:"Y",
    12:"up",
    13:"right",
    14:"down",
    15:"left",

}

class JoystickListenerNode(Node):
    def __init__(self, joystick_tolerance = 0.01):
        super().__init__('go2_joystick_listener')
        
        self.joystick_sub = self.create_subscription(
            WirelessController, "/wirelesscontroller", self.listener_callback, 1
        )
        self.received_msg = None
        self.msg_dict = None
        self.joystick_tolerance = joystick_tolerance
        self.received_new_msg = False
    
    def listener_callback(self, msg:WirelessController):
        
        if self.received_msg:
            if msg.keys == self.received_msg.keys and \
                np.abs(msg.lx - self.received_msg.lx) <= self.joystick_tolerance and \
                np.abs(msg.ly - self.received_msg.ly) <= self.joystick_tolerance and \
                np.abs(msg.rx - self.received_msg.rx) <= self.joystick_tolerance and \
                np.abs(msg.ry - self.received_msg.ry) <= self.joystick_tolerance:
                return
        if msg.keys != 0:
            print(f"[JoystickListener]: received new key {msg.keys}")
        self.msg_dict = {
            "lx": msg.lx,
            "ly": msg.ly,
            "rx": msg.rx,
            "ry": msg.ry,
            "keys": msg.keys,
        }
        self.received_msg = msg
        self.received_new_msg = True

    def get_dict(self):
        return copy.deepcopy(self.msg_dict)


class JoystickListener(mp.Process):
    def __init__(
            self,
            shm_manager,
            verbose = True,
        ):
        super().__init__(name="Go2Arx5Controller")
        
        self.shm_manager = shm_manager

        # build queue
        example = dict()
        example["lx"] = 0.0
        example['ly'] = 0.0
        example['rx'] = 0.0
        example['ry'] = 0.0
        example['keys'] = int(0)

        queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256,
        )
        print("queue created")
        self.ready_event = mp.Event()
        self.queue = queue

        # Should be initialized in the subprocess
        self.sub_node: Go2Arx5Listener
        self.verbose = verbose
    
    # ========= launch method ===========
    def start(self):
        super().start()
        # rclpy.init()
        if self.verbose:
            print(f"[JoystickListener] Subprocess started")

    
    def stop(self):
        # rclpy.shutdown()
        self.join()
        if self.verbose:
            print(f"[JoystickListener] Subprocess terminated")
    
     
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= receive APIs =============
    def get_msg(self, k=None, out=None):
        if k is None:
            return self.queue.get(out=out)
        else:
            return self.queue.get_k(k=k,out=out)

    def get_key_events(self):
        if self.queue.qsize() == 0:
            return []
        all_msgs = self.queue.get_all()
        pressed_keys = []
        prev_key_msg = 0
        key_msgs = all_msgs["keys"]
        for key_msg in key_msgs:
            if key_msg != prev_key_msg:
                # new key pressed
                for order in KEY_ORDER.keys():
                    if (key_msg >> order) & 1 and not (prev_key_msg >> order) & 1:
                        pressed_keys.append(KEY_ORDER[order])
            prev_key_msg = key_msg
        return pressed_keys
                


    # ========= main loop (only for listener update) ============
    def run(self):
        # rclpy.init()
        self.sub_node = JoystickListenerNode()
        try:
            while rclpy.ok():
                rclpy.spin_once(self.sub_node)
                if self.sub_node.received_new_msg:
                    # update queue
                    msg_dict = self.sub_node.get_dict()
                    if msg_dict is not None:
                        self.queue.put(msg_dict)
                    self.sub_node.received_new_msg = False
        except:
            print(echo_exception())

def echo_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    # Extract unformatted traceback
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    # Print line of code where the exception occurred

    return "".join(tb_lines)


if __name__ == "__main__":
    # Test subscriber output
    with SharedMemoryManager() as shm_manager:
        with JoystickListener(shm_manager=shm_manager) as joystick_listener:
            while True:
                keys = joystick_listener.get_key_events()
                print(keys)
                time.sleep(1)