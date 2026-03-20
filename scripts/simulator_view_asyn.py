"""
MuJoCo机器人控制脚本
实现位置和力矩混合控制，读取传感器数据
"""
import mujoco
import mujoco.viewer
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading
import rclpy
from rclpy.node import Node
# import message_filters
from bodyctrl_msgs.msg import MotorCtrl, MotorStatus,MotorStatusMsg, CmdMotorCtrl, Imu, Euler
from queue import Queue, Full
from pynput import keyboard
from scipy.spatial.transform import Rotation
import os
from elastic_band import ElasticBand
import argparse


def quaternion_to_euler_scipy(w, x, y, z, degrees=False):
    """
    使用SciPy进行四元数转欧拉角
    """
    # 创建旋转对象 (注意顺序: [x, y, z, w])
    rotation = Rotation.from_quat([x, y, z, w])

    # 转换为欧拉角 (顺序: 'xyz' 对应 roll, pitch, yaw)
    euler_angles = rotation.as_euler('xyz', degrees=degrees)

    return euler_angles[0], euler_angles[1], euler_angles[2]


class RobotSimulator:

    def __init__(self, model_path, node: Node | None = None, debug=False, robot_config='full'):
        self.debug = debug
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 初始化控制模式 (0=(位置控制(仅上半身)和力控都存在), 1=全身仅力矩控制)
        self.control_mode = 1

        # 记录数据用于绘图
        record_length = 10000
        self.time_history = deque(maxlen=record_length)
        self.joint_pos_history = deque(maxlen=record_length)
        self.joint_vel_history = deque(maxlen=record_length)
        self.joint_torque_history = deque(maxlen=record_length)
        self.imu_orient_history = deque(maxlen=record_length)
        self.imu_accel_history = deque(maxlen=record_length)
        self.imu_pos_history = deque(maxlen=record_length)

        # 根据机器人配置设置关节映射
        if robot_config == '21':
            # 21自由度配置（没有手腕关节）
            self.motor_id_to_joint = {
                1: 'head_roll_joint',
                2: 'head_pitch_joint',
                3: 'head_yaw_joint',
                11: 'shoulder_pitch_l_joint',
                12: 'shoulder_roll_l_joint',
                13: 'shoulder_yaw_l_joint',
                14: 'elbow_pitch_l_joint',

                21: 'shoulder_pitch_r_joint',
                22: 'shoulder_roll_r_joint',
                23: 'shoulder_yaw_r_joint',
                24: 'elbow_pitch_r_joint',

                33: 'waist_yaw_joint',
                51: 'hip_pitch_l_joint',
                52: 'hip_roll_l_joint',
                53: 'hip_yaw_l_joint',
                54: 'knee_pitch_l_joint',
                55: 'ankle_pitch_l_joint',
                56: 'ankle_roll_l_joint',
                61: 'hip_pitch_r_joint',
                62: 'hip_roll_r_joint',
                63: 'hip_yaw_r_joint',
                64: 'knee_pitch_r_joint',
                65: 'ankle_pitch_r_joint',
                66: 'ankle_roll_r_joint'
            }
            
            #双臂关节名称
            self.arm_joint_names = [
                'shoulder_pitch_l_joint', 'shoulder_roll_l_joint',
                'shoulder_yaw_l_joint', 'elbow_pitch_l_joint', 
                'shoulder_pitch_r_joint', 'shoulder_roll_r_joint',
                'shoulder_yaw_r_joint', 'elbow_pitch_r_joint', 
            ]
            #腰部关节名
            self.waist_joint_names = [
                'waist_yaw_joint', 
            ]
        else:
            # 全配置（包含手腕关节）
            self.motor_id_to_joint = {
                1: 'head_roll_joint',
                2: 'head_pitch_joint',
                3: 'head_yaw_joint',
                11: 'shoulder_pitch_l_joint',
                12: 'shoulder_roll_l_joint',
                13: 'shoulder_yaw_l_joint',
                14: 'elbow_pitch_l_joint',
                15: 'elbow_yaw_l_joint',
                16: 'wrist_pitch_l_joint',
                17: 'wrist_roll_l_joint',
                21: 'shoulder_pitch_r_joint',
                22: 'shoulder_roll_r_joint',
                23: 'shoulder_yaw_r_joint',
                24: 'elbow_pitch_r_joint',
                25: 'elbow_yaw_r_joint',
                26: 'wrist_pitch_r_joint',
                27: 'wrist_roll_r_joint',
                # 31: 'waist_yaw_joint',
                # 32: 'waist_roll_joint',
                # 33: 'waist_pitch_joint',
                33: 'waist_yaw_joint',
                32: 'waist_roll_joint',
                31: 'waist_pitch_joint',
                51: 'hip_pitch_l_joint',
                52: 'hip_roll_l_joint',
                53: 'hip_yaw_l_joint',
                54: 'knee_pitch_l_joint',
                55: 'ankle_pitch_l_joint',
                56: 'ankle_roll_l_joint',
                61: 'hip_pitch_r_joint',
                62: 'hip_roll_r_joint',
                63: 'hip_yaw_r_joint',
                64: 'knee_pitch_r_joint',
                65: 'ankle_pitch_r_joint',
                66: 'ankle_roll_r_joint'
            }
            
            #双臂关节名称
            self.arm_joint_names = [
                'shoulder_pitch_l_joint', 'shoulder_roll_l_joint',
                'shoulder_yaw_l_joint', 'elbow_pitch_l_joint', 'elbow_yaw_l_joint',
                'wrist_pitch_l_joint', 'wrist_roll_l_joint',
                'shoulder_pitch_r_joint', 'shoulder_roll_r_joint',
                'shoulder_yaw_r_joint', 'elbow_pitch_r_joint', 'elbow_yaw_r_joint',
                'wrist_pitch_r_joint', 'wrist_roll_r_joint'
            ]
            #腰部关节名
            self.waist_joint_names = [
                'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'
            ]
        
        # 关节到电机ID的映射
        self.joint_to_motor_id = {
            joint: id
            for id, joint in self.motor_id_to_joint.items()
        }

        #腿部关节名称
        self.leg_joint_names = [
            'hip_roll_l_joint', 'hip_pitch_l_joint', 'hip_yaw_l_joint',
            'knee_pitch_l_joint', 'ankle_pitch_l_joint', 'ankle_roll_l_joint',
            'hip_roll_r_joint', 'hip_pitch_r_joint', 'hip_yaw_r_joint',
            'knee_pitch_r_joint', 'ankle_pitch_r_joint', 'ankle_roll_r_joint'
        ]
        # 获取传感器/执行器索引
        self.get_all_robot_info()

        self.sim_view_lock = threading.Lock()
        
        # 升降带
        self.elastic_band = ElasticBand(enable=True)
        # 根据配置设置弹性带连接的链接
        if robot_config == '21':
            self.band_attached_link = self.model.body("waist_yaw_link").id
        else:
            self.band_attached_link = self.model.body("waist_pitch_link").id

        # 启动可视化器
        self.paused = False
        # self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer = mujoco.viewer.launch_passive(
        self.model, self.data, key_callback=self.elastic_band.MujuocoKeyCallback
        )
        self.view_dt = 0.03
        # 线程控制变量
        self.joint_cmd_lock = threading.Lock()
        self.stop_event = threading.Event()

        #创建ROS相关话题
        self.node = node
        if self.node:
            self.node.create_subscription(CmdMotorCtrl, 'arm/cmd_ctrl',
                                          self.joint_cmd_callback, 10)
            self.node.create_subscription(CmdMotorCtrl, 'leg/cmd_ctrl',
                                          self.joint_cmd_callback, 10)
            self.node.create_subscription(CmdMotorCtrl, 'waist/cmd_ctrl',
                                          self.joint_cmd_callback, 10)
            #创建发布者
            self.arm_status_pub = self.node.create_publisher(
                MotorStatusMsg, 'arm/status', 20)
            self.leg_status_pub = self.node.create_publisher(
                MotorStatusMsg, 'leg/status', 20)
            self.waist_status_pub = self.node.create_publisher(
                MotorStatusMsg, 'waist/status', 10)
            self.imu_status_pub = self.node.create_publisher(
                Imu, 'imu/status', 10)
        # 传感器返回的机器人状态
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_torques = {}
        self.imu_orientation = None
        self.imu_position = None
        self.imu_angular_velocity = None
        self.imu_linear_velocity = None
        self.imu_linear_acceleration = None
        self.imu_magnetometer = None

        # 关节命令
        self.raw_joint_commands = Queue()
        self.joint_commands = {}
        self.last_raw_joint_commands = {}

        # 发布线程相关
        self.pub_thread = None
        self.pub_thread_lock = threading.Lock()
        self.pub_thread_running = False
        self.last_sensor_data = None
        self.sensor_data_lock = threading.Lock()

        # 初始化控制器
        self.init_controller()
    def init_controller(self):
        if self.control_mode == 1:
            # 将位置控制器的增益设为0
            for idx in self.pos_actuator_indices_map.values():
                self.model.actuator_gainprm[idx, 0] = 0  # kp = 0
                self.model.actuator_biasprm[idx, 1] = 0  # kv = 0 (biasprm[1]通常是kv)
            self.control_mode = 1
            print("切换到力矩控制模式：位置控制器已禁用")
    def switch_to_torque_control(self):
        """切换到力矩控制：禁用位置控制器"""
        if self.control_mode == 0:
            # 将位置控制器的增益设为0
            for idx in self.pos_actuator_indices_map.values():
                self.model.actuator_gainprm[idx, 0] = 0  # kp = 0
                self.model.actuator_biasprm[idx, 1] = 0  # kv = 0 (biasprm[1]通常是kv)
            self.control_mode = 1
            print("切换到力矩控制模式：位置控制器已禁用")
    
    def switch_to_position_torque_control(self):
        """切换到位置力控混合控制模式：启用位置控制器"""
        if self.control_mode == 1:
            # 恢复位置控制器的原始增益
            for idx in self.pos_actuator_indices_map.values():
                self.model.actuator_gainprm[idx] = self.original_gains[idx]
                self.model.actuator_biasprm[idx] = self.original_biases[idx]
            self.control_mode = 0
            print("切换到位置力控混合控制模式：位置控制器已启用")

    def start_keyboard_listener(self):
        """使用 pynput 监听键盘"""

        def on_press(key):
            try:
                if key == keyboard.Key.num_lock:
                    self.paused = not self.paused
                    print(f"仿真已{'暂停' if self.paused else '继续'}")
                if key == keyboard.Key.end:
                    if self.control_mode == 1:
                        self.switch_to_position_torque_control()
                    else:
                        self.switch_to_torque_control()
            except Exception as e:
                print(f"键盘监听错误: {e}")

        def listen():
            with keyboard.Listener(on_press=on_press) as listener:
                while not self.stop_event.is_set():
                    time.sleep(0.1)
                listener.stop()

        thread = threading.Thread(target=listen, daemon=True)
        thread.start()

    def start_publishing(self):
        """启动发布线程"""
        if not self.node:
            return
        with self.pub_thread_lock:
            if not self.pub_thread_running:
                self.pub_thread_running = True
                self.pub_thread = threading.Thread(target=self._publish_thread,
                                                   daemon=True)
                self.pub_thread.start()

    def stop_publishing(self):
        """停止发布线程"""
        with self.pub_thread_lock:
            if self.pub_thread_running:
                self.pub_thread_running = False
                if self.pub_thread:
                    self.pub_thread.join(timeout=1.0)

    def _publish_thread(self):
        """发布状态的独立线程"""
        pub_freq = 400  # 100Hz发布频率
        pub_interval = 1.0 / pub_freq
        sensor_data = None
        while self.pub_thread_running and not self.stop_event.is_set():
            start_time = time.perf_counter()

            # 获取最新的传感器数据
            with self.sensor_data_lock:
                if self.last_sensor_data is not None:
                    sensor_data = self.last_sensor_data.copy()
            # 为每个关节组发布状态
            self._publish_joint_group(self.arm_joint_names,
                                      self.arm_status_pub, sensor_data)
            self._publish_joint_group(self.waist_joint_names,
                                      self.waist_status_pub, sensor_data)
            self._publish_joint_group(self.leg_joint_names,
                                      self.leg_status_pub, sensor_data)
            # 发布Imu
            self._publish_imu_status(self.imu_status_pub, sensor_data)
            # 控制发布频率
            elapsed = time.perf_counter() - start_time
            sleep_time = pub_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _publish_imu_status(self, publisher, sensor_data):
        if publisher and sensor_data:
            orientaion = sensor_data['imu_orientation']
            # position = sensor_data['imu_position']
            imu_angular_velocity = sensor_data['imu_angular_velocity']
            imu_linear_acceleration = sensor_data['imu_linear_acceleration']
            imu_msg: Imu = Imu()
            imu_msg.header.stamp = self.node.get_clock().now().to_msg()
            imu_msg.orientation.x = orientaion[1]
            imu_msg.orientation.y = orientaion[2]
            imu_msg.orientation.z = orientaion[3]
            imu_msg.orientation.w = orientaion[0]
            # 四元数转换为欧拉角
            r, p, y = quaternion_to_euler_scipy(*list(orientaion))
            imu_msg.euler = Euler()
            imu_msg.euler.roll = r
            imu_msg.euler.pitch = p
            imu_msg.euler.yaw = y
            imu_msg.angular_velocity.x = imu_angular_velocity[0]
            imu_msg.angular_velocity.y = imu_angular_velocity[1]
            imu_msg.angular_velocity.z = imu_angular_velocity[2]
            imu_msg.linear_acceleration.x = imu_linear_acceleration[0]
            imu_msg.linear_acceleration.y = imu_linear_acceleration[1]
            imu_msg.linear_acceleration.z = imu_linear_acceleration[2]

            try:
                publisher.publish(imu_msg)
            except Exception as e:
                print(f"发布 imu 状态时出错: {e}")

    def _publish_joint_group(self, joint_names, publisher, sensor_data):
        """发布指定关节组的状态"""
        if publisher and sensor_data:
            joint_positions = sensor_data['joint_positions']
            joint_velocities = sensor_data['joint_velocities']
            joint_torques = sensor_data['joint_torques']
            #构造关节状态消息
            motor_status_msg = MotorStatusMsg()
            motor_status_msg.header.stamp = self.node.get_clock().now().to_msg()
            for joint_name in joint_names:
                if joint_name in joint_positions and joint_name in joint_velocities:
                    motor_status = self.construct_motor_status(
                        joint_name, joint_positions[joint_name],
                        joint_velocities[joint_name], joint_torques[joint_name])
                    motor_status_msg.status.append(motor_status)
            try:
                publisher.publish(motor_status_msg)
            except Exception as e:
                print(f"发布状态{motor_status_msg}时出错: {e}")

    def get_all_robot_info(self):
        # 获取关节索引
        self.joint_names = self.arm_joint_names + self.waist_joint_names + self.leg_joint_names
        # 获取执行器索引
        self.pos_actuator_names = [f"pos_{name}" for name in self.joint_names]
        self.motor_actuator_names = [
            f"motor_{name}" for name in self.joint_names
        ]

        self.pos_actuator_indices_map = {}
        self.motor_actuator_indices_map = {}

        for name in self.pos_actuator_names:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                    name)
            self.pos_actuator_indices_map[name] = idx

        for name in self.motor_actuator_names:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                    name)
            self.motor_actuator_indices_map[name] = idx

        # 获取执行器增益
        self.original_gains = self.model.actuator_gainprm.copy()
        self.original_biases = self.model.actuator_biasprm.copy()

        # 获取传感器索引
        self.joint_pos_sensor_names = [
            f"{name}_pos" for name in self.joint_names
        ]
        self.joint_vel_sensor_names = [
            f"{name}_vel" for name in self.joint_names
        ]
        self.joint_torque_sensor_names = [
            f"{name}_torque" for name in self.joint_names
        ]

        self.joint_pos_sensor_indices_map = {}
        self.joint_vel_sensor_indices_map = {}
        self.joint_torque_sensor_indices_map = {}

        for name in self.joint_pos_sensor_names:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR,
                                    name)
            self.joint_pos_sensor_indices_map[name] = idx

        for name in self.joint_vel_sensor_names:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR,
                                    name)
            self.joint_vel_sensor_indices_map[name] = idx

        for name in self.joint_torque_sensor_names:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR,
                                    name)
            self.joint_torque_sensor_indices_map[name] = idx

        # IMU传感器索引
        self.imu_orient_idx = mujoco.mj_name2id(self.model,
                                                mujoco.mjtObj.mjOBJ_SENSOR,
                                                'orientation')
        self.imu_orient_address = self.model.sensor_adr[self.imu_orient_idx]
        self.imu_pos_idx = mujoco.mj_name2id(self.model,
                                             mujoco.mjtObj.mjOBJ_SENSOR,
                                             'position')
        self.imu_pos_address = self.model.sensor_adr[self.imu_pos_idx]
        self.imu_gyro_idx = mujoco.mj_name2id(self.model,
                                              mujoco.mjtObj.mjOBJ_SENSOR,
                                              'angular-velocity')
        self.imu_gyro_address = self.model.sensor_adr[self.imu_gyro_idx]
        self.imu_vel_idx = mujoco.mj_name2id(self.model,
                                             mujoco.mjtObj.mjOBJ_SENSOR,
                                             'linear-velocity')
        self.imu_vel_address = self.model.sensor_adr[self.imu_vel_idx]
        self.imu_accel_idx = mujoco.mj_name2id(self.model,
                                               mujoco.mjtObj.mjOBJ_SENSOR,
                                               'linear-acceleration')
        self.imu_accel_address = self.model.sensor_adr[self.imu_accel_idx]
        self.imu_mag_idx = mujoco.mj_name2id(self.model,
                                             mujoco.mjtObj.mjOBJ_SENSOR,
                                             'magnetometer')
        self.imu_mag_address = self.model.sensor_adr[self.imu_mag_idx]

    def pvd_controller(self, joint_name, pos_d, spd_d, tor_d, kp, kd):
        """
        PVD控制器实现函数，用于计算关节的输出扭矩
        
        参数:
        joint_name: 关节名称，用于获取关节状态
        pos_d: 期望位置
        spd_d: 期望速度
        tor_d: 期望扭矩
        kp: 位置比例增益系数
        kd: 速度比例增益系数
        
        返回值:
        output_torque: 计算得到的输出扭矩值
        """

        status = self.get_status(joint_name)
        pos_a = status['position']
        spd_a = status['velocity']
        output_torque = kp * (pos_d - pos_a) + kd * (spd_d - spd_a) + tor_d
        return output_torque

    def joint_cmd_callback(self, msg: CmdMotorCtrl):
        cmd: MotorCtrl = None
        # 每一组命令全部接受完再释放锁
        with self.joint_cmd_lock:
            current_time = self.node.get_clock().now().to_msg()
            msg_time = msg.header.stamp
            time_diff = (current_time.sec - msg_time.sec) * 1000000000 + (current_time.nanosec - msg_time.nanosec)
            time_diff_ms = time_diff / 1000000.0
            if time_diff_ms > 100:
                print(f"[WARNING] Time difference: {time_diff_ms} ms, abandon this command")
                return
                
            for cmd in msg.cmds:
                try:
                    self.raw_joint_commands.put_nowait(cmd)
                except Full:
                    print(f"raw joint command 队列已满")
                    self.raw_joint_commands.get_nowait()
                    self.raw_joint_commands.put_nowait(cmd)

    def calc_motor_cmd(self, joint_name, pos, spd, tor, kp, kd):
        # joint_name = self.motor_id_to_joint[name_id]
        # 计算下发力矩
        target_torque = self.pvd_controller(joint_name, pos, spd, tor, kp, kd)
        self.joint_commands[joint_name] = target_torque
        if self.debug:
            print(f"{joint_name} torque : {target_torque}\n")

    def set_motor_cmd(self):
        with self.joint_cmd_lock:
            while not self.raw_joint_commands.empty():
                cmd: MotorCtrl = self.raw_joint_commands.get_nowait()
                name = cmd.name
                pos = cmd.pos
                spd = cmd.spd
                tor = cmd.tor
                kp = cmd.kp
                kd = cmd.kd
                joint_name = self.motor_id_to_joint[name]
                self.last_raw_joint_commands[joint_name] = { "pos": pos, "spd": spd, "tor": tor, "kp": kp, "kd": kd}
                # self.calc_motor_cmd(name, pos, spd, tor, kp, kd)


        for joint_name in self.joint_names :
            if joint_name in self.last_raw_joint_commands:
                pos = self.last_raw_joint_commands[joint_name].get("pos")
                spd = self.last_raw_joint_commands[joint_name].get("spd")
                tor = self.last_raw_joint_commands[joint_name].get("tor")
                kp = self.last_raw_joint_commands[joint_name].get("kp")
                kd = self.last_raw_joint_commands[joint_name].get("kd")
                self.calc_motor_cmd(joint_name, pos, spd, tor, kp, kd)
                motor_name = f"motor_{joint_name}"
                motor_index = self.motor_actuator_indices_map[motor_name]
                if motor_index != -1:
                    # 确保电机存在且已收到命令
                    self.data.ctrl[motor_index] = self.joint_commands[joint_name]
    def viewer_thread(self):
        try:
            while self.viewer.is_running() and not self.stop_event.is_set():
                with self.sim_view_lock:
                    self.viewer.sync()
                time.sleep(self.view_dt)
        except Exception as e:
            print(f"Viewer thread error: {e}")
        finally:
            print("Viewer thread stopped")

    def read_sensors(self):
        """读取所有传感器数据"""
        # 读取关节传感器数据

        for name in self.joint_names:
            pos_sensor_name = f"{name}_pos"
            vel_sensor_name = f"{name}_vel"
            torque_sensor_name = f"{name}_torque"
            pos_idx = self.joint_pos_sensor_indices_map[pos_sensor_name]
            vel_idx = self.joint_vel_sensor_indices_map[vel_sensor_name]
            torque_idx = self.joint_torque_sensor_indices_map[
                torque_sensor_name]

            self.joint_positions[name] = self.data.sensordata[pos_idx]
            self.joint_velocities[name] = self.data.sensordata[vel_idx]
            self.joint_torques[name] = self.data.sensordata[torque_idx]

        # 读取IMU数据
        self.imu_orientation = self.data.sensordata[self.imu_orient_address:self.
                                                    imu_orient_address + 4]
        self.imu_position = self.data.sensordata[self.
                                                 imu_pos_address:self.imu_pos_address +
                                                 3]
        self.imu_angular_velocity = self.data.sensordata[self.imu_gyro_address:self
                                                         .imu_gyro_address + 3]
        self.imu_linear_velocity = self.data.sensordata[self.imu_vel_address:self.
                                                        imu_vel_address + 3]
        self.imu_linear_acceleration = self.data.sensordata[
            self.imu_accel_address:self.imu_accel_address + 3]
        self.imu_magnetometer = self.data.sensordata[self.imu_mag_address:self.
                                                     imu_mag_address + 3]
        # 更新传感器数据供发布线程使用
        plot_data = {
            'joint_positions': np.array(list(self.joint_positions.values())),
            'joint_velocities': np.array(list(self.joint_velocities.values())),
            'joint_torques': np.array(list(self.joint_torques.values())),
            'imu_orientation': self.imu_orientation,
            'imu_position': self.imu_position,
            'imu_angular_velocity': self.imu_angular_velocity,
            'imu_linear_velocity': self.imu_linear_velocity,
            'imu_linear_acceleration': self.imu_linear_acceleration,
            'imu_magnetometer': self.imu_magnetometer
        }
        with self.sensor_data_lock:
            self.last_sensor_data = {
                'joint_positions': self.joint_positions.copy(),
                'joint_velocities': self.joint_velocities.copy(),
                'joint_torques': self.joint_torques.copy(),
                'imu_orientation': self.imu_orientation,
                'imu_position': self.imu_position,
                'imu_angular_velocity': self.imu_angular_velocity,
                'imu_linear_velocity': self.imu_linear_velocity,
                'imu_linear_acceleration': self.imu_linear_acceleration,
                'imu_magnetometer': self.imu_magnetometer
            }

        return plot_data

    def construct_motor_status(self, joint_name, pos, speed, torque=None):
        """
        构建电机状态消息
        
        Args:
            joint_name (str): 关节名称
            pos (float): 位置值
            speed (float): 速度值
            
        Returns:
            MotorStatus: 电机状态消息
        """
        motor_status = MotorStatus()
        motor_status.name = self.joint_to_motor_id.get(joint_name, int())
        motor_status.pos = pos
        motor_status.speed = speed
        motor_status.current = torque

        return motor_status

    def get_status(self, joint_name):
        # Todo: 处理还未收到传感器状态的情况
        with self.sensor_data_lock:
            joint_positions = self.last_sensor_data['joint_positions'].copy()
            joint_velocities = self.last_sensor_data['joint_velocities'].copy()
            joint_torques = self.last_sensor_data['joint_torques'].copy()
        pos = joint_positions[joint_name]
        vel = joint_velocities[joint_name]
        torque = joint_torques[joint_name]
        return {'position': pos, 'velocity': vel, 'torque': torque}

    def simulate_thread(self):
        """运行控制循环"""
        # 控制参数
        time_step = 0
        control_frequency = 100  # 100Hz控制频率
        sensor_frequency = 100  # 100Hz传感器读取频率
        control_interval = 1 / control_frequency

        target_time = 0
        # last_control_time = 0
        try:
            while self.viewer.is_running() and not self.stop_event.is_set():
                step_start = time.perf_counter()

                target_time += self.model.opt.timestep
                with self.sim_view_lock:
                    # 读取传感器数据
                    if int(1.0 / self.model.opt.timestep * target_time) % int(
                            1.0 / self.model.opt.timestep /
                            sensor_frequency) == 0:
                        sensor_data = self.read_sensors()
                        # 存储数据用于绘图
                        self.time_history.append(target_time)
                        self.joint_pos_history.append(
                            sensor_data['joint_positions'].copy())
                        self.joint_vel_history.append(
                            sensor_data['joint_velocities'].copy())
                        self.joint_torque_history.append(
                            sensor_data['joint_torques'].copy())
                        self.imu_orient_history.append(
                            sensor_data['imu_orientation'].copy())
                        self.imu_accel_history.append(
                            sensor_data['imu_linear_acceleration'].copy())
                        self.imu_pos_history.append(
                            sensor_data['imu_position'].copy())
                        # 打印部分传感器数据
                        if time_step % 100 == 0 and self.debug:
                            print(f"Time: {target_time:.3f}s")
                            print(
                                f"Control Mode: {'Position' if self.control_mode == 0 else 'Torque'}"
                            )
                            print(
                                f"Joint 0 Pos: {sensor_data['joint_positions'][0]:.3f}, "
                                f"Vel: {sensor_data['joint_velocities'][0]:.3f}, "
                                f"Torque: {sensor_data['joint_torques'][0]:.3f}"
                            )
                            print(
                                f"IMU Orientation: [{sensor_data['imu_orientation'][0]:.3f}, "
                                f"{sensor_data['imu_orientation'][1]:.3f}, "
                                f"{sensor_data['imu_orientation'][2]:.3f}, "
                                f"{sensor_data['imu_orientation'][3]:.3f}]")
                            print(
                                f"IMU Acceleration: [{sensor_data['imu_linear_acceleration'][0]:.3f}, "
                                f"{sensor_data['imu_linear_acceleration'][1]:.3f}, "
                                f"{sensor_data['imu_linear_acceleration'][2]:.3f}]"
                            )
                            print("-" * 50)
                    # 添加虚拟力
                    self.add_virtual_force()
                    # 控制命令写入
                    self.set_motor_cmd()
                    # 前进仿真一步
                    if not self.paused:
                        mujoco.mj_step(self.model, self.data)
                    else:
                        mujoco.mj_forward(self.model, self.data)
                time_step += 1

                # 控制仿真速度
                time_until_next_step = self.model.opt.timestep - (time.perf_counter() -
                                                                  step_start)
                print(f'sim timestep: {time_until_next_step}')
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        except Exception as e:
            print(f"Simulation thread error: {e}")
        finally:
            print("Simulation thread stopped")

    def add_virtual_force(self):
        if self.elastic_band.enable:
            self.data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.Advance(
            self.data.qpos[0:3], self.data.qvel[0:3]
        )
    def plot_sensor_data(self):
        """绘制传感器数据"""
        if len(self.time_history) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sensor Data')

        time_array = np.array(list(self.time_history))

        # 绘制关节位置
        if len(self.joint_pos_history) > 0:
            pos_array = np.array(list(self.joint_pos_history))
            axes[0, 0].plot(time_array, pos_array[:, 0], label='Joint 0')
            axes[0, 0].plot(time_array, pos_array[:, 1], label='Joint 1')
            axes[0, 0].set_title('Joint Positions')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Position (rad)')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

        # 绘制关节速度
        if len(self.joint_vel_history) > 0:
            vel_array = np.array(list(self.joint_vel_history))
            axes[0, 1].plot(time_array, vel_array[:, 0], label='Joint 0')
            axes[0, 1].plot(time_array, vel_array[:, 1], label='Joint 1')
            axes[0, 1].set_title('Joint Velocities')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Velocity (rad/s)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # 绘制IMU方向
        if len(self.imu_orient_history) > 0:
            orient_array = np.array(list(self.imu_orient_history))
            axes[1, 0].plot(time_array, orient_array[:, 0], label='w')
            axes[1, 0].plot(time_array, orient_array[:, 1], label='x')
            axes[1, 0].plot(time_array, orient_array[:, 2], label='y')
            axes[1, 0].plot(time_array, orient_array[:, 3], label='z')
            axes[1, 0].set_title('IMU Orientation (Quaternion)')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Quaternion')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # # 绘制IMU加速度
        # if len(self.imu_accel_history) > 0:
        #     accel_array = np.array(list(self.imu_accel_history))
        #     axes[1, 1].plot(time_array, accel_array[:, 0], label='X')
        #     axes[1, 1].plot(time_array, accel_array[:, 1], label='Y')
        #     axes[1, 1].plot(time_array, accel_array[:, 2], label='Z')
        #     axes[1, 1].set_title('IMU Linear Acceleration')
        #     axes[1, 1].set_xlabel('Time (s)')
        #     axes[1, 1].set_ylabel('Acceleration (m/s²)')
        #     axes[1, 1].legend()
        #     axes[1, 1].grid(True)

        # 绘制IMU位置
        if len(self.imu_pos_history) > 0:
            pos_array = np.array(list(self.imu_pos_history))
            axes[1, 1].plot(time_array, pos_array[:, 0], label='X')
            axes[1, 1].plot(time_array, pos_array[:, 1], label='Y')
            axes[1, 1].plot(time_array, pos_array[:, 2], label='Z')
            axes[1, 1].set_title('IMU Linear Position')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Position (m)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def start(self):
        from threading import Thread
        viewer_thread = Thread(target=self.viewer_thread)
        sim_thread = Thread(target=self.simulate_thread)

        # 设置为守护线程，确保主线程结束时它们也会结束
        viewer_thread.daemon = True
        sim_thread.daemon = True

        try:
            # 设置初始姿态
            # self.set_keyframe("zero")  # 或者 "zero_height" 或 "zero_standup"， 'zero'
            # print("机器人已设置为zero姿态")

            self.start_keyboard_listener()
            viewer_thread.start()
            sim_thread.start()
            # 启动发布线程
            self.start_publishing()
            # 主线程等待，直到收到 KeyboardInterrupt
            while viewer_thread.is_alive() and sim_thread.is_alive():
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nReceived KeyboardInterrupt, stopping simulation...")
        finally:
            # 设置停止事件
            self.stop_event.set()

            # 等待线程结束
            viewer_thread.join(timeout=1.0)
            sim_thread.join(timeout=1.0)
            # 停止发布线程
            self.stop_publishing()
            # 关闭 viewer
            try:
                self.viewer.close()
            except:
                pass

            # 绘制图表
            if self.debug:
                print("Plotting sensor data...")
                self.plot_sensor_data()
            print("Simulation finished")

    def set_keyframe(self, key_name):
        """
        将机器人设置为指定的关键帧姿态
        
        参数:
            key_name: 关键帧名称 (例如 "zero", "zero_height", "zero_standup")
        """
        # 查找关键帧的索引
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY,
                                   key_name)

        if key_id == -1:
            print(f"关键帧 '{key_name}' 未找到")
            return False
        # # # 将关键帧的qpos值复制到当前状态
        self.data.qpos[:] = self.model.key_qpos[key_id]

        # 将关键帧的ctrl值复制到控制向量
        if self.model.key_ctrl is not None:
            self.data.ctrl[:] = self.model.key_ctrl[key_id]

        mujoco.mj_forward(self.model, self.data)

        # 更新系统状态
        # mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        return True

    def get_available_keyframes(self):
        """
        获取所有可用的关键帧名称
        """
        keyframes = []
        for i in range(self.model.nkey):
            # 获取关键帧名称
            key_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY,
                                         i)
            if key_name:
                keyframes.append(key_name)
        return keyframes

def _spin_wrapper(node):
    """ROS2 spin wrapper for graceful shutdown."""
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)
    except Exception as e:
        print(f"Error in ROS2 spin thread: {e}")
    finally:
        print("ROS2 spin thread terminated.")


def main():
    # 设置模型注册表
    model_registry = {
        'evt2': '../resources/evt2/urdf/evt2.xml'
    }

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MuJoCo机器人仿真器')
    parser.add_argument('--model', '-m', type=str, default='evt2',
                        choices=list(model_registry.keys()),
                        help='要加载的机器人模型')
    parser.add_argument('--config', '-c', type=str, default='full',
                        choices=['full', '21'],
                        help='机器人配置 (full=完整配置, 21=21自由度配置)')

    args = parser.parse_args()

    # 设置模型路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_registry[args.model])
    model_path = os.path.abspath(model_path)
    
    print(f"正在加载模型: {model_path}")
    print(f"机器人配置: {args.config}")

    # 创建控制器
    rclpy.init()
    node = rclpy.create_node('mujoco_simulator_dex')
    spin_thread = threading.Thread(target=_spin_wrapper,
                                   args=(node, ),
                                   daemon=True,
                                   name="ROS2_Spin_Thread")
    spin_thread.start()
    simulator = RobotSimulator(model_path, node, debug=False, robot_config=args.config)

    # 运行仿真
    simulator.start()


if __name__ == "__main__":
    main()