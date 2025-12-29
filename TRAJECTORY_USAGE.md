# 轨迹生成功能使用说明

## 概述

双层GMPC控制器现在支持两种运行模式：
1. **固定点跟踪模式**（默认）：末端执行器跟踪固定目标位置和姿态
2. **螺旋轨迹跟踪模式**：末端执行器沿螺旋轨迹运动（参考 Jacobian.cpp）

## 代码位置

### 1. 轨迹生成函数
位置：[src/gmpc_dual_layer.cpp](src/gmpc_dual_layer.cpp) (第 40-100 行)

```cpp
// 辅助函数：计算四元数轨迹
static Eigen::Vector4d calculateQuaternionTrajectory(...)

// 螺旋轨迹生成器（与 Jacobian.cpp 一致）
static Eigen::MatrixXd calculateDesiredState(double t_start, double dt, int horizonLength)
{
    double t_end = 30.0;     // 总时间 30秒
    double radius = 0.4;     // 螺旋半径 0.4m
    double height = 0.2;     // 螺旋高度 0.2m
    double turns = 0.6;      // 旋转圈数
    Eigen::Vector4d q0(0.6, 0, 0, 0.8);  // 初始姿态
    
    // ... 生成 13 维状态 [qw qx qy qz px py pz wx wy wz vx vy vz]
}
```

### 2. 控制器中的轨迹调用
位置：[src/cartesian_impedance_controller.cpp](src/cartesian_impedance_controller.cpp) (第 237-290 行)

```cpp
if (use_trajectory_) {
    // 螺旋轨迹模式
    double t_in_period = std::fmod(t_total_, 30.0);
    // 根据当前时间计算期望位置、速度、姿态
    xd0.v(4) = x;   // 位置
    xd0.v(10) = x_d; // 速度
} else {
    // 固定点模式
    xd0.v(4) = position_d_(0);  // 使用固定目标
}
```

## 如何切换运行模式

### 方法 1：修改代码（重新编译）

**步骤：**

1. 打开 [include/serl_franka_controllers/cartesian_impedance_controller.h](include/serl_franka_controllers/cartesian_impedance_controller.h)

2. 找到第 55 行附近：
   ```cpp
   bool use_trajectory_ = false;    // 是否使用轨迹跟踪（false=固定点）
   ```

3. 修改为：
   ```cpp
   bool use_trajectory_ = true;     // 启用螺旋轨迹跟踪
   ```

4. 重新编译：
   ```bash
   cd ~/catkin_ws
   catkin_make --pkg serl_franka_controllers
   source devel/setup.bash
   ```

5. 启动控制器：
   ```bash
   roslaunch serl_franka_controllers impedance.launch robot_ip:=YOUR_ROBOT_IP
   ```

### 方法 2：添加ROS参数（推荐）

**未来改进建议**：可以将 `use_trajectory_` 作为 ROS 参数从 launch 文件加载，这样无需重新编译即可切换。

在 `impedance.launch` 中添加：
```xml
<param name="use_trajectory" value="true"/>  <!-- 或 false -->
```

在 `cartesian_impedance_controller.cpp` 的 `init()` 函数中读取：
```cpp
node_handle.param<bool>("use_trajectory", use_trajectory_, false);
```

## 轨迹参数调整

### 螺旋轨迹参数
编辑 [src/cartesian_impedance_controller.cpp](src/cartesian_impedance_controller.cpp) 第 245-250 行：

```cpp
double T = 30.0;         // 总周期（秒）- 完成一个螺旋的时间
double radius = 0.4;     // 螺旋半径（米）- 调大会增加工作空间
double height = 0.2;     // 螺旋高度（米）- 沿z轴上升的总距离
double turns = 0.6;      // 旋转圈数 - 调大会增加旋转速度
```

**注意事项：**
- `radius` 太大可能超出机械臂工作空间
- `turns` 太大会导致角速度过快
- 修改后需要重新编译

### 固定点参数
当 `use_trajectory_ = false` 时，目标点由以下决定：
- **初始值**：启动时的末端位置（参见 `starting()` 函数）
- **运行中**：通过 ROS topic 发布新目标（参见 `equilibriumPoseCallback()`）

```bash
# 发布新的目标位置
rostopic pub /equilibrium_pose geometry_msgs/PoseStamped ...
```

## 轨迹生成原理

### 参考实现
螺旋轨迹生成完全参考 [Jacobian.cpp](Jacobian.cpp) 第 1376-1415 行：

```cpp
// 位置：圆周运动 + 垂直上升
x = radius * cos(2π * turns * t / t_end)
y = radius * sin(2π * turns * t / t_end)
z = z0 + height * t / t_end

// 速度：位置对时间求导
vx = -radius * (2π * turns) / t_end * sin(...)
vy =  radius * (2π * turns) / t_end * cos(...)
vz = height / t_end
```

### 姿态处理
- **固定姿态**：四元数 `q0(0.6, 0, 0, 0.8)` 保持不变
- **角速度**：`w0_ref(0, 0, 0)` = 零向量

如需旋转姿态，可修改 `w0_ref` 或使用 `calculateQuaternionTrajectory()` 函数。

## 调试信息

### 轨迹模式日志
启用轨迹跟踪后，终端会每 2 秒输出：
```
Trajectory mode | t=5.23 | pos=[0.387, 0.124, 0.935] | vel=[-0.041, 0.098, 0.007]
```

### GMPC状态日志
GMPC活跃时每秒输出：
```
GMPC active | tau_norm: 12.345 | position_error: [0.001, 0.002, 0.001]
```

## 常见问题

**Q1: 轨迹不运动？**
- 检查 `use_trajectory_` 是否设为 `true`
- 检查终端是否有 "Trajectory mode" 日志
- 确认 GMPC 求解成功（无 "GMPC solver failed" 警告）

**Q2: 机械臂抖动？**
- 轨迹参数可能过于激进（`radius` 或 `turns` 太大）
- 检查 GMPC 权重参数（`Q_`, `R_` 在 gmpc_dual_layer.cpp）
- 降低速度：减小 `turns` 或增大 `T`

**Q3: 如何添加自定义轨迹？**
1. 在 `cartesian_impedance_controller.cpp` 的轨迹生成部分添加新的计算逻辑
2. 参考 Jacobian.cpp 中的注释掉的轨迹（固定点、直线）
3. 确保生成正确的 13 维状态向量 `[qw qx qy qz px py pz wx wy wz vx vy vz]`

## 与 Jacobian.cpp 的对应关系

| Jacobian.cpp | gmpc_dual_layer.cpp | 说明 |
|--------------|---------------------|------|
| `calculateDesiredState()` (行1376) | `calculateDesiredState()` (行67) | 轨迹生成函数 |
| `calculateQuaternionTrajectory()` (行1560) | `calculateQuaternionTrajectory()` (行45) | 四元数计算 |
| `computeMPC()` 中调用 (行780) | `cartesian_impedance_controller.cpp` 调用 (行250) | 集成方式 |

**关键区别：**
- Jacobian.cpp：单层GMPC，轨迹生成在 `computeMPC()` 内部
- gmpc_dual_layer.cpp：双层GMPC，轨迹生成在 controller 中调用

## 下一步建议

1. **添加更多轨迹类型**：
   - 直线轨迹（参考 Jacobian.cpp 注释部分）
   - 圆形轨迹
   - 自定义路径点轨迹

2. **ROS参数化**：
   - 将 `use_trajectory_`, `radius`, `height` 等作为 ROS 参数
   - 支持运行时动态切换

3. **轨迹可视化**：
   - 发布期望轨迹到 RViz
   - 对比实际末端位置和期望位置

4. **外部轨迹源**：
   - 订阅 ROS topic 接收外部生成的轨迹
   - 支持实时轨迹规划器
