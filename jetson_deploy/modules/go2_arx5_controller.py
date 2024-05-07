import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile


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

    def publish_target_traj(self, pose_traj, gripper_traj, timestamps):
        pass

    def get_state(self):


class Go2Arx5Controller:
    def __init__(
            self,
            shm_manager,
            verbose=True,
        ):
        self.shm_manager = shm_manager
        self.verbose = verbose
        self.node = Go2Arx5Node()
        rclpy.init()

    # ========= receive APIs =============
    def get_state(self):
        rclpy.spin_once(self.node)

    
    
    