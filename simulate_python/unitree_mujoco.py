import time
import threading
from threading import Thread
from pathlib import Path
import argparse
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rosgraph_msgs.msg import Clock

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config

locker = threading.Lock()

current_dir = Path(__file__).resolve().parent
xml_path = current_dir / config.ROBOT_SCENE

mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
mj_data = mujoco.MjData(mj_model)

# -------------------------
# ROS 2 /clock publisher
# -------------------------
class MujocoClockPublisher(Node):
    def __init__(self):
        super().__init__("mujoco_clock_publisher")

        # /clock 常见 QoS：reliable + transient_local，depth=1
        # 这样 late-join 的订阅者也能拿到最后一条 clock（更稳）
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(Clock, "/clock", qos)

        # 可选：用于观测发布频率
        self._last_print_t = 0.0

    def publish_sim_time(self, sim_time_sec: float):
        # sim_time_sec: float seconds
        sec = int(sim_time_sec)
        nanosec = int((sim_time_sec - sec) * 1e9)
        if nanosec < 0:
            nanosec = 0

        msg = Clock()
        msg.clock.sec = sec
        msg.clock.nanosec = nanosec
        self.pub.publish(msg)

        # 可选日志（每 2 秒打印一次）
        # now_wall = time.time()
        # if now_wall - self._last_print_t > 2.0:
        #     self.get_logger().info(f"Publishing /clock = {sec}.{nanosec:09d}")
        #     self._last_print_t = now_wall


_clock_node = None  # type: MujocoClockPublisher | None


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def SimulationThread(ratio=1.0):
    global mj_data, mj_model, _clock_node

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )

        mujoco.mj_step(mj_model, mj_data)

        # 发布仿真时间到 /clock
        # MuJoCo 的仿真时间在 mj_data.time（秒）
        if _clock_node is not None:
            _clock_node.publish_sim_time(float(mj_data.time))

        locker.release()

        time_until_next_step = mj_model.opt.timestep / ratio - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


def RosSpinThread():
    # 后台 spin，保证 ROS 2 通信稳定
    rclpy.spin(_clock_node)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Simulation speed ratio (e.g. 0.5 = slower, 2.0 = faster)"
    )
    args = parser.parse_args()
    ratio = args.ratio
    
    # ROS 2 init
    rclpy.init()
    _clock_node = MujocoClockPublisher()

    # 启动 ROS spin 线程
    ros_thread = Thread(target=RosSpinThread, daemon=True)
    ros_thread.start()

    # 你的仿真/显示线程
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread, args=(ratio,))

    viewer_thread.start()
    sim_thread.start()

    # 等待结束
    sim_thread.join()
    viewer_thread.join()

    # ROS 2 shutdown
    _clock_node.destroy_node()
    rclpy.shutdown()
