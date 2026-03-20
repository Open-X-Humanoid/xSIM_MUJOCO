# MuJoCo DEX 机器人仿真项目

本项目是基于 MuJoCo 物理引擎的 DEX 机器人仿真环境，用于开发和测试机器人控制算法。

## 项目结构

```
.
├── README.md                # 项目说明文档
├── resources                # 机器人模型文件
│   ├── evt2
└── scripts                      # Python 脚本目录
    ├── convert_xml.py           # URDF 到 XML 转换工具
    ├── elastic_band.py          # 弹性带控制器
    └── simulator_view_asyn.py   # 异步仿真器（支持 ROS2）
```

## 主要功能

### 1. 机器人模型
- 包含完整的 DEX 机器人模型，具有多个自由度
- 支持腿部、腰部和手臂关节控制
- 集成 IMU 传感器和关节传感器

### 2. 控制系统
- 支持位置控制和力矩控制模式
- 实现 PVD（位置-速度-力矩）混合控制器
- 提供传感器数据读取接口

### 3. 仿真功能
- 实时物理仿真
- 3D 可视化显示
- 数据记录和绘图功能
- ROS2 集成支持（通过 simulator_view_asyn.py）

## 安装依赖

```bash
# 建议使用虚拟环境
pip install mujoco numpy matplotlib pynput
```
bodyctrl_msg 位于xmigcs/lib/下
```bash
sudo dpkg -i ros-jazzy*.deb
```
## 使用方法

### 异步仿真（支持 ROS2）

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=your_domain_id
python scripts/simulator_view_asyn.py -m evt2(机器人名称)
```

## 主要脚本说明

### simulator_view_asyn.py
异步版本的机器人仿真器，支持：
- 多线程运行
- ROS2 集成
- 键盘控制
- 实时数据发布

### elastic_band.py
弹性带控制器

## 机器人关节结构

DEX 机器人包含以下关节组：

1. **腿部关节**：
   - hip_pitch_l_joint, hip_roll_l_joint, hip_yaw_l_joint
   - knee_pitch_l_joint, ankle_pitch_l_joint, ankle_roll_l_joint
   - 右腿对应关节

2. **腰部关节**：
   - waist_yaw_joint, waist_roll_joint, waist_pitch_joint

3. **手臂关节**：
   - shoulder_pitch_l_joint, shoulder_roll_l_joint, shoulder_yaw_l_joint
   - elbow_pitch_l_joint, elbow_yaw_l_joint, wrist_pitch_l_joint, wrist_roll_l_joint
   - 右臂对应关节

## 控制接口

通过修改 `data.ctrl[]` 数组来控制关节力矩，通过读取 `data.sensordata[]` 获取传感器数据。

## 键盘控制

在仿真运行时：
- end：切换力控、位控
- numlock: 暂停、继续
- up: 升起机器人
- down: 降下机器人
- ESC: 释放机器人，可以自由运动


## 数据记录

仿真器会自动记录以下数据用于后续分析和可视化：
- 关节位置、速度、力矩
- IMU 方向、加速度、位置数据

## 注意事项

1. 确保 MuJoCo 模型文件路径正确
2. 如需 ROS2 支持，请先启动 ROS2 环境
3. 仿真速度受系统性能影响，可通过调整 timestep 参数优化