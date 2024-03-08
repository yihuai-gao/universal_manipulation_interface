from typing import Any, Optional, Union, cast
import zmq
from enum import IntEnum, auto
import numpy.typing as npt
import numpy as np
import sys
import traceback


def echo_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    # Extract unformatted traceback
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    # Print line of code where the exception occurred

    return "".join(tb_lines)


CTRL_DT = 0.005


class Arx5Client:
    def __init__(
        self,
        zmq_ip: str,
        zmq_port: int,
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{zmq_ip}:{zmq_port}")
        self.zmq_ip = zmq_ip
        self.zmq_port = zmq_port
        self.latest_state: dict[str, Union[npt.NDArray[np.float64], float]]
        print(
            f"Arx5Client is connected to {self.zmq_ip}:{self.zmq_port}. Fetching state..."
        )
        self.get_state()
        print(f"Initial state fetched")

    def send_recv(self, msg: dict[str, Any]):
        try:
            self.socket.send_pyobj(msg)
            reply_msg = self.socket.recv_pyobj()
            return reply_msg
        except KeyboardInterrupt:
            print("Arx5Client: KeyboardInterrupt. connection is re-established.")
            return {"cmd": msg["cmd"], "data": "KeyboardInterrupt"}
        except zmq.error.ZMQError:
            # Usually happens when the process is interrupted before receiving reply
            print("Arx5Client: ZMQError. connection is re-established.")
            print(echo_exception())
            self.socket.close()
            del self.socket
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(f"tcp://{self.zmq_ip}:{self.zmq_port}")
            return {"cmd": msg["cmd"], "data": "ZMQError"}

    def get_state(self):
        reply_msg = self.send_recv({"cmd": "GET_STATE", "data": None})
        assert reply_msg["cmd"] == "GET_STATE"
        assert isinstance(reply_msg["data"], dict)
        if reply_msg["data"] == "KeyboardInterrupt" or reply_msg["data"] == "ZMQError":
            return self.latest_state
        state = cast(
            dict[str, Union[npt.NDArray[np.float64], float]], reply_msg["data"]
        )
        self.latest_state = state
        return state

    def set_ee_pose(
        self, pose_6d: npt.NDArray[np.float64], gripper_pos: Union[float, None] = None
    ):
        reply_msg = self.send_recv(
            {
                "cmd": "SET_EE_POSE",
                "data": {"ee_pose": pose_6d, "gripper_pos": gripper_pos},
            }
        )
        assert reply_msg["cmd"] == "SET_EE_POSE"
        if reply_msg["data"] == "KeyboardInterrupt" or reply_msg["data"] == "ZMQError":
            return self.latest_state
        if type(reply_msg["data"]) != dict:
            raise ValueError(f"Error: {reply_msg['data']}")
        state = cast(
            dict[str, Union[npt.NDArray[np.float64], float]], reply_msg["data"]
        )
        self.latest_state = state
        return state

    def reset_to_home(self):
        reply_msg = self.send_recv({"cmd": "RESET_TO_HOME", "data": None})
        assert reply_msg["cmd"] == "RESET_TO_HOME"
        if reply_msg["data"] != "OK":
            raise ValueError(f"Error: {reply_msg['data']}")

    def set_to_damping(self):
        reply_msg = self.send_recv({"cmd": "SET_TO_DAMPING", "data": None})
        assert reply_msg["cmd"] == "SET_TO_DAMPING"
        if reply_msg["data"] != "OK":
            raise ValueError(f"Error: {reply_msg['data']}")

    def get_gain(self):
        reply_msg = self.send_recv({"cmd": "GET_GAIN", "data": None})
        assert reply_msg["cmd"] == "GET_GAIN"
        if type(reply_msg["data"]) != dict:
            raise ValueError(f"Error: {reply_msg['data']}")
        return cast(dict[str, Union[npt.NDArray[np.float64], float]], reply_msg["data"])

    def set_gain(self, gain: dict[str, Union[npt.NDArray[np.float64], float]]):
        reply_msg = self.send_recv({"cmd": "SET_GAIN", "data": gain})
        assert reply_msg["cmd"] == "SET_GAIN"
        if reply_msg["data"] != "OK":
            raise ValueError(f"Error: {reply_msg['data']}")

    @property
    def timestamp(self):
        timestamp = self.latest_state["timestamp"]
        return cast(float, timestamp)

    @property
    def ee_pose(self):
        ee_pose = self.latest_state["ee_pose"]
        return cast(npt.NDArray[np.float64], ee_pose)

    @property
    def joint_pos(self):
        joint_pos = self.latest_state["joint_pos"]
        return cast(npt.NDArray[np.float64], joint_pos)

    @property
    def joint_vel(self):
        joint_vel = self.latest_state["joint_vel"]
        return cast(npt.NDArray[np.float64], joint_vel)

    @property
    def joint_torque(self):
        joint_torque = self.latest_state["joint_torque"]
        return cast(npt.NDArray[np.float64], joint_torque)

    @property
    def gripper_pos(self):
        gripper_pos = self.latest_state["gripper_pos"]
        return cast(float, gripper_pos)

    @property
    def gripper_vel(self):
        gripper_vel = self.latest_state["gripper_vel"]
        return cast(float, gripper_vel)

    @property
    def gripper_torque(self):
        gripper_torque = self.latest_state["gripper_torque"]
        return cast(float, gripper_torque)

    def __del__(self):
        self.socket.close()
        self.context.term()
        print("Arx5Client is closed")
