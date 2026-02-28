# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import argparse
import select
import sys
import termios
import threading
import time
import tty
from collections import deque

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from PIL import Image
from piper_msgs.msg import PosCmd
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header

from openpi_client import image_tools
from openpi_client import websocket_client_policy

CAMERA_NAMES = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
# INSTRUCTION = "Pick up a pack of tissue paper and put it in the plastic basket."
# INSTRUCTION = "Pick up the beverage and put it in the plastic basket."
# INSTRUCTION = "Pick up the trash and throw it in the trash bin."
# INSTRUCTION = "Take a napkin and put it on the table."
INSTRUCTION = "Fold the towel."

observation_window = None

# Global flag for interactive mode
interactive_mode_flag = False


def check_keyboard_input():
    """Check if a key was pressed without blocking."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def handle_interactive_mode():
    """
    Handle interactive mode when space is pressed.
    Returns: 'continue' to resume, 'reset' to restart from beginning
    """
    print("\n" + "=" * 50)
    print("INTERACTIVE MODE")
    print("  'c' - Continue running")
    print("  'r' - Reset to starting point and restart")
    print("  'q' - Quit/Stop")
    print("=" * 50)

    while True:
        key = sys.stdin.read(1).lower()
        if key == "c":
            print("Continuing...")
            return "continue"
        elif key == "r":
            print("Restarting...")
            return "reset"
        elif key == "q":
            print("Stopping...")
            return "quit"


class OpenpiInference:
    def __init__(
        self,
        host="0.0.0.0",
        port=8000,
    ) -> None:

        # build client to connect server policy
        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)

    def predict_action(self, payload) -> np.ndarray:
        images = [payload["top"], payload["left"], payload["right"]]
        images = [image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)) for img in images]
        images = [img.transpose(2, 0, 1) for img in images]

        observation = {
            "state": payload["state"],
            "images": {
                "cam_high": images[0],
                "cam_left_wrist": images[1],
                "cam_right_wrist": images[2],
            },
            "prompt": payload["instruction"],
        }

        response = self.client.infer(observation)
        unnormalized_actions = response["actions"]  # chunk, D

        return unnormalized_actions


def get_config(args):
    config = {
        "episode_len": args.max_publish_step,
        "state_dim": 14,
        "chunk_size": args.chunk_size,
        "camera_names": CAMERA_NAMES,
    }
    return config


def quat_to_RPY(quat):
    """
    Convert quaternion in (x, y, z, w) order to roll, pitch, yaw (intrinsic).

    Args:
        quat: quaternion as [x, y, z, w]

    Returns:
        tuple: (roll, pitch, yaw) in the specified units
    """

    # Create rotation object from quaternion (scipy expects [x, y, z, w] format)
    rot = R.from_quat([quat.x, quat.y, quat.z, quat.w])

    # Convert to Euler angles (intrinsic R→P→Y = 'xyz')
    euler_angles = rot.as_euler("xyz", degrees=False)

    return euler_angles


# Get the observation from the ROS topic
def get_ros_observation(args, ros_operator):
    rate = rospy.Rate(args.publish_rate)
    print_flag = True

    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail when get_ros_observation")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (
            img_front,
            img_left,
            img_right,
            img_front_depth,
            img_left_depth,
            img_right_depth,
            puppet_arm_left,
            puppet_arm_right,
            endpose_left,
            endpose_right,
            robot_base,
        ) = result
        # print(f"sync success when get_ros_observation")
        return (
            img_front,
            img_left,
            img_right,
            puppet_arm_left,
            puppet_arm_right,
            endpose_left,
            endpose_right,
        )


# Update the observation window buffer
def update_observation_window(args, config, ros_operator):
    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)

        # Append the first dummy image
        observation_window.append(
            {
                "qpos": None,
                "images": {
                    config["camera_names"][0]: None,
                    config["camera_names"][1]: None,
                    config["camera_names"][2]: None,
                },
                "endpose": None,
            }
        )

    img_front, img_left, img_right, puppet_arm_left, puppet_arm_right, endpose_left, endpose_right = (
        get_ros_observation(args, ros_operator)
    )

    qpos = np.concatenate(
        (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)),
        axis=0,
    )

    left_pos = endpose_left.pose.position
    left_rpy = quat_to_RPY(endpose_left.pose.orientation)
    left_gripper = puppet_arm_left.position[-1]
    endpose_left = np.array([left_pos.x, left_pos.y, left_pos.z, left_rpy[0], left_rpy[1], left_rpy[2], left_gripper])

    right_pos = endpose_right.pose.position
    right_rpy = quat_to_RPY(endpose_right.pose.orientation)
    right_gripper = puppet_arm_right.position[-1]
    endpose_right = np.array(
        [right_pos.x, right_pos.y, right_pos.z, right_rpy[0], right_rpy[1], right_rpy[2], right_gripper]
    )

    endpose = np.concatenate((endpose_left, endpose_right), axis=0)

    observation_window.append(
        {
            "qpos": qpos,
            "images": {
                config["camera_names"][0]: img_front,
                config["camera_names"][1]: img_right,
                config["camera_names"][2]: img_left,
            },
            "endpose": endpose,
        }
    )


def inference_fn(args, config, policy):
    global observation_window

    # print(f"Start inference_thread_fn: t={t}")
    while True and not rospy.is_shutdown():
        start_time = time.time()

        # fetch images in sequence [front, right, left]
        image_arrs = [
            observation_window[-1]["images"][config["camera_names"][0]],
            observation_window[-1]["images"][config["camera_names"][1]],
            observation_window[-1]["images"][config["camera_names"][2]],
        ]

        if args.ctrl_type == "joint":
            # state: Abs Joint 14dim
            state = observation_window[-1]["qpos"]
        elif args.ctrl_type == "eef":
            # state: Abs EEF 14dim
            state = observation_window[-1]["endpose"]

        payload = {
            "top": image_arrs[0],
            "right": image_arrs[1],
            "left": image_arrs[2],
            "instruction": INSTRUCTION,
            "state": state,
        }

        actions = policy.predict_action(payload)

        print(f"Model inference time: {(time.time() - start_time)*1000:.3f} ms")

        return actions


# Main loop for the manipulation task
def model_inference(args, config, ros_operator):
    policy = OpenpiInference(host=args.host, port=args.port)

    max_publish_step = config["episode_len"]
    chunk_size = config["chunk_size"]

    # Initialize position of the puppet arm
    # left0 = [-0.44922745, 0.5082583, -1.0291806, 0.05915684, 0.79456437, -0.11905055, 0.092]
    # right0 = [0.40189838, 0.5602516, -1.1094662, -0.03117958, 0.84036225, 0.01538969, 0.092]
    left0 = [
        -0.00133514404296875,
        0.00209808349609375,
        0.01583099365234375,
        -0.032616615295410156,
        -0.00286102294921875,
        0.00095367431640625,
        3.557830810546875,
    ]
    right0 = [
        -0.00133514404296875,
        0.00438690185546875,
        0.034523963928222656,
        -0.053597450256347656,
        -0.00476837158203125,
        -0.00209808349609375,
        3.557830810546875,
    ]
    # left0 = [0.02, 0.01, 0, 0, 0.34, 0.16, 0]
    # right0 = [0, 0.32, -0.36, 0, 0.24, 0, 0.1]

    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Press enter to continue")
    ros_operator.puppet_arm_publish_continuous(left0, right0)

    # Set terminal to raw mode for non-blocking keyboard input
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    print("\nPress SPACE during inference to enter interactive mode")

    try:
        # Inference loop
        while True and not rospy.is_shutdown():
            # The current time step
            t = 0
            rate = rospy.Rate(args.publish_rate)

            action_buffer = np.zeros([chunk_size, config["state_dim"]])

            while t < max_publish_step and not rospy.is_shutdown():
                # Check for keyboard input (space to enter interactive mode)
                key = check_keyboard_input()
                if key == " ":
                    result = handle_interactive_mode()
                    if result == "reset":
                        # Reset to starting position
                        ros_operator.puppet_arm_publish_continuous(left0, right0)
                        input("Press enter to continue")
                        break  # Break inner loop to restart
                    elif result == "quit":
                        return  # Exit the function entirely
                    # 'continue' just resumes the loop

                # Update observation window
                update_observation_window(args, config, ros_operator)

                # When coming to the end of the action chunk
                if t % chunk_size == 0:
                    # Start inference
                    action_buffer = inference_fn(args, config, policy).copy()

                raw_action = action_buffer[t % chunk_size]
                act = raw_action

                if args.ctrl_type == "joint":
                    left_action = act[:7]
                    right_action = act[7:14]

                    if right_action[6] < 0.05:
                        right_action[6] -= 0.01
                    elif right_action[6] > 0.05:
                        right_action[6] += 0.01

                    ros_operator.puppet_arm_publish(left_action, right_action)
                elif args.ctrl_type == "eef":
                    left_action = act[:7]
                    right_action = act[7:14]
                    ros_operator.endpose_publish(left_action, right_action)

                if args.use_robot_base:
                    vel_action = act[14:16]
                    ros_operator.robot_base_publish(vel_action)
                rate.sleep()

                t += 1
                print("Published Step", t)
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


# ROS operator class
class RosOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.endpose_right_deque = None
        self.endpose_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.endpose_left_publisher = None
        self.endpose_right_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.endpose_left_deque = deque()
        self.endpose_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()  # Set timestep
        joint_state_msg.name = [
            "joint0",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def endpose_publish(self, left, right):
        endpose_msg = PosCmd()
        endpose_msg.x, endpose_msg.y, endpose_msg.z = left[:3]
        endpose_msg.roll, endpose_msg.pitch, endpose_msg.yaw = left[3:6]
        endpose_msg.gripper = left[6]
        self.endpose_left_publisher.publish(endpose_msg)

        endpose_msg.x, endpose_msg.y, endpose_msg.z = right[:3]
        endpose_msg.roll, endpose_msg.pitch, endpose_msg.yaw = right[3:6]
        endpose_msg.gripper = right[6]
        self.endpose_right_publisher.publish(endpose_msg)

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # Set the timestep
            joint_state_msg.name = [
                "joint0",
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ]
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            print("puppet_arm_publish_continuous:", step)
            rate.sleep()

    def get_frame(self):
        if (
            len(self.img_left_deque) == 0
            or len(self.img_right_deque) == 0
            or len(self.img_front_deque) == 0
            or (
                self.args.use_depth_image
                and (
                    len(self.img_left_depth_deque) == 0
                    or len(self.img_right_depth_deque) == 0
                    or len(self.img_front_depth_deque) == 0
                )
            )
        ):
            return False
        if self.args.use_depth_image:
            frame_time = min(
                [
                    self.img_left_deque[-1].header.stamp.to_sec(),
                    self.img_right_deque[-1].header.stamp.to_sec(),
                    self.img_front_deque[-1].header.stamp.to_sec(),
                    self.img_left_depth_deque[-1].header.stamp.to_sec(),
                    self.img_right_depth_deque[-1].header.stamp.to_sec(),
                    self.img_front_depth_deque[-1].header.stamp.to_sec(),
                ]
            )
        else:
            frame_time = min(
                [
                    self.img_left_deque[-1].header.stamp.to_sec(),
                    self.img_right_deque[-1].header.stamp.to_sec(),
                    self.img_front_deque[-1].header.stamp.to_sec(),
                ]
            )

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (
            len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time
        ):
            return False
        if self.args.use_depth_image and (
            len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time
        ):
            return False
        if self.args.use_depth_image and (
            len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time
        ):
            return False
        if self.args.use_robot_base and (
            len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time
        ):
            return False
        if len(self.endpose_left_deque) == 0 or self.endpose_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.endpose_right_deque) == 0 or self.endpose_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), "passthrough")

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), "passthrough")

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), "passthrough")

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        while self.endpose_left_deque[0].header.stamp.to_sec() < frame_time:
            self.endpose_left_deque.popleft()
        endpose_left = self.endpose_left_deque.popleft()

        while self.endpose_right_deque[0].header.stamp.to_sec() < frame_time:
            self.endpose_right_deque.popleft()
        endpose_right = self.endpose_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), "passthrough")

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), "passthrough")

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), "passthrough")

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (
            img_front,
            img_left,
            img_right,
            img_front_depth,
            img_left_depth,
            img_right_depth,
            puppet_arm_left,
            puppet_arm_right,
            endpose_left,
            endpose_right,
            robot_base,
        )

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def endpose_left_callback(self, msg):
        if len(self.endpose_left_deque) >= 2000:
            self.endpose_left_deque.popleft()
        self.endpose_left_deque.append(msg)

    def endpose_right_callback(self, msg):
        if len(self.endpose_right_deque) >= 2000:
            self.endpose_right_deque.popleft()
        self.endpose_right_deque.append(msg)

    def init_ros(self):
        rospy.init_node("joint_state_publisher", anonymous=True)
        rospy.Subscriber(
            self.args.img_left_topic,
            Image,
            self.img_left_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.args.img_right_topic,
            Image,
            self.img_right_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.args.img_front_topic,
            Image,
            self.img_front_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        if self.args.use_depth_image:
            rospy.Subscriber(
                self.args.img_left_depth_topic,
                Image,
                self.img_left_depth_callback,
                queue_size=1000,
                tcp_nodelay=True,
            )
            rospy.Subscriber(
                self.args.img_right_depth_topic,
                Image,
                self.img_right_depth_callback,
                queue_size=1000,
                tcp_nodelay=True,
            )
            rospy.Subscriber(
                self.args.img_front_depth_topic,
                Image,
                self.img_front_depth_callback,
                queue_size=1000,
                tcp_nodelay=True,
            )
        rospy.Subscriber(
            self.args.puppet_arm_left_topic,
            JointState,
            self.puppet_arm_left_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.args.puppet_arm_right_topic,
            JointState,
            self.puppet_arm_right_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.args.robot_base_topic,
            Odometry,
            self.robot_base_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.args.endpose_left_topic,
            PoseStamped,
            self.endpose_left_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.args.endpose_right_topic,
            PoseStamped,
            self.endpose_right_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )

        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(
            self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10
        )
        self.endpose_left_publisher = rospy.Publisher(self.args.endpose_left_cmd_topic, PosCmd, queue_size=10)
        self.endpose_right_publisher = rospy.Publisher(self.args.endpose_right_cmd_topic, PosCmd, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_publish_step",
        action="store",
        type=int,
        help="Maximum number of action publishing steps",
        default=10000,
        required=False,
    )
    parser.add_argument(
        "--img_front_topic",
        action="store",
        type=str,
        help="img_front_topic",
        default="/camera_f/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_left_topic",
        action="store",
        type=str,
        help="img_left_topic",
        default="/camera_l/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_right_topic",
        action="store",
        type=str,
        help="img_right_topic",
        default="/camera_r/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_front_depth_topic",
        action="store",
        type=str,
        help="img_front_depth_topic",
        default="/camera_f/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_left_depth_topic",
        action="store",
        type=str,
        help="img_left_depth_topic",
        default="/camera_l/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_right_depth_topic",
        action="store",
        type=str,
        help="img_right_depth_topic",
        default="/camera_r/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_left_cmd_topic",
        action="store",
        type=str,
        help="puppet_arm_left_cmd_topic",
        default="/master/joint_left",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_right_cmd_topic",
        action="store",
        type=str,
        help="puppet_arm_right_cmd_topic",
        default="/master/joint_right",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_left_topic",
        action="store",
        type=str,
        help="puppet_arm_left_topic",
        default="/puppet/joint_left",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_right_topic",
        action="store",
        type=str,
        help="puppet_arm_right_topic",
        default="/puppet/joint_right",
        required=False,
    )
    parser.add_argument(
        "--endpose_left_cmd_topic",
        action="store",
        type=str,
        help="endpose_left_cmd_topic",
        default="/pos_cmd_left",
        required=False,
    )
    parser.add_argument(
        "--endpose_right_cmd_topic",
        action="store",
        type=str,
        help="endpose_right_cmd_topic",
        default="/pos_cmd_right",
        required=False,
    )
    parser.add_argument(
        "--endpose_left_topic",
        action="store",
        type=str,
        default="/puppet/end_pose_left",
        required=False,
    )
    parser.add_argument(
        "--endpose_right_topic",
        action="store",
        type=str,
        default="/puppet/end_pose_right",
        required=False,
    )
    parser.add_argument(
        "--robot_base_topic",
        action="store",
        type=str,
        help="robot_base_topic",
        default="/odom_raw",
        required=False,
    )
    parser.add_argument(
        "--robot_base_cmd_topic",
        action="store",
        type=str,
        help="robot_base_topic",
        default="/cmd_vel",
        required=False,
    )
    parser.add_argument(
        "--use_robot_base",
        action="store_true",
        help="Whether to use the robot base to move around",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--publish_rate",
        action="store",
        type=int,
        help="The rate at which to publish the actions",
        default=30,
        required=False,
    )
    parser.add_argument(
        "--chunk_size",
        action="store",
        type=int,
        help="Action chunk size",
        default=50,
        required=False,
    )
    parser.add_argument(
        "--arm_steps_length",
        action="store",
        type=float,
        help="The maximum change allowed for each joint per timestep",
        default=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.2],
        required=False,
    )
    parser.add_argument(
        "--use_depth_image",
        action="store_true",
        help="Whether to use depth images",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--ctrl_type",
        type=str,
        choices=["joint", "eef"],
        help="Control type for the robot arm",
        default="joint",
    )
    parser.add_argument(
        "--host",
        action="store",
        type=str,
        help="Websocket server host",
        default="127.0.0.1",
        required=False,
    )
    parser.add_argument(
        "--port",
        action="store",
        type=int,
        help="Websocket server port",
        default=8000,
        required=False,
    )

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    config = get_config(args)
    model_inference(args, config, ros_operator)


if __name__ == "__main__":
    main()
