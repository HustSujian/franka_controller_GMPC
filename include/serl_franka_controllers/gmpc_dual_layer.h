// ====================================================================================
// gmpc_dual_layer.h - 双层GMPC控制器公共接口头文件
// ====================================================================================
// 
// 【头文件作用】
// 1. 定义公共数据结构（参数、输入、期望轨迹）
// 2. 声明控制器接口类 GMPCDualLayer（供外部调用）
// 3. 前向声明内部实现类 DualLayerGMPC（隐藏实现细节）
// 
// 【为什么没有声明内部函数】
// 本头文件采用 Pimpl（指针实现）设计模式：
//   - GMPCDualLayer：公共接口类，只声明外部需要的接口函数
//   - DualLayerGMPC：内部实现类，在.cpp中完整定义，包含所有内部函数
// 
// 优点：
//   1. 编译隔离：修改内部实现不需要重新编译所有依赖此头文件的代码
//   2. 接口简洁：外部只看到必要的接口，不暴露复杂的内部算法细节
//   3. 依赖隔离：OSQP等内部求解器细节不泄露到头文件
//   4. 二进制兼容：内部实现改变不影响ABI（应用程序二进制接口）
// 
// 【使用方式】
// 在控制器中：
//   #include <serl_franka_controllers/gmpc_dual_layer.h>
//   
//   GMPCDualLayer gmpc_;           // 创建GMPC对象
//   GMPCParams params;             // 设置参数
//   params.Nt = 10;
//   gmpc_.setParams(params);
//   
//   DesiredState13 xd0;            // 期望状态
//   Eigen::Matrix<double,7,1> tau; // 输出力矩
//   gmpc_.computeTauMPC(..., &tau);
// 
// 【内部函数在哪里】
// 所有内部函数（buildPrimaryQP, solveOSQP等）在 gmpc_dual_layer.cpp 的 DualLayerGMPC 类中实现
// 不需要在此头文件声明，因为外部不直接调用它们
// ====================================================================================

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <OsqpEigen/OsqpEigen.h>
#include <array>
#include <vector>
#include <cstdint>
#include <memory>

namespace serl_franka_controllers {

// ===============================
// GMPC参数结构体 - 存储所有可调参数
// 用途：控制器调用 setParams() 传入此结构配置GMPC行为
// 参考：7dof-双层.txt MATLAB实现 + Jacobian.cpp验证的参数
// ===============================
struct GMPCParams {
  // dimensions
  int Nx = 12;
  int Nu = 7;
  int Nt = 10;

  // discretization
  double dt = 0.001;

  // bounds
  Eigen::Matrix<double, 12, 1> xmin = Eigen::Matrix<double, 12, 1>::Constant(-10.0);
  Eigen::Matrix<double, 12, 1> xmax = Eigen::Matrix<double, 12, 1>::Constant( 10.0);

  Eigen::Matrix<double, 7, 1> umin;
  Eigen::Matrix<double, 7, 1> umax;

  // costs
  Eigen::Matrix<double, 12, 12> Q;
  Eigen::Matrix<double, 7, 7>   R;
  Eigen::Matrix<double, 12, 12> P;

  // second layer weights (from your MATLAB)
  double w_smooth = 1e-8;
  double w_null   = 1e-6;
  Eigen::Matrix<double,7,7> R_smooth = Eigen::Matrix<double,7,7>::Identity();
  Eigen::Matrix<double,7,7> R_null   = (Eigen::Matrix<double,7,7>() <<
      0.1,0,0,0,0,0,0,
      0,1.0,0,0,0,0,0,
      0,0,1.0,0,0,0,0,
      0,0,0,1.0,0,0,0,
      0,0,0,0,0.1,0,0,
      0,0,0,0,0,0.1,0,
      0,0,0,0,0,0,0.1).finished();

  // second layer: deviation bound (delta_deviation)
  double delta_deviation = 0.001;   // 
  double lambda_dls      = 0.01;    // DLS base

  // performance tolerance (alpha_tolerance) —  MATLAB 有，但这里先留接口
  double alpha_tolerance = 0.95;

  GMPCParams() {
    umax << 50, 50, 50, 28, 28, 28, 28;
    umin = -umax;

    // Jacobian.cpp: Q_ / R_ 的量级
    Q.setZero();
    Q.block<3,3>(0,0) = 20.0 * Eigen::Matrix3d::Identity();     // rotation err
    Q.block<3,3>(3,3) = 20.0 * Eigen::Matrix3d::Identity();     // translation err
    Q.block<6,6>(6,6) = 800.0 * Eigen::Matrix<double,6,6>::Identity(); // twist (or vel) err part

    // Q = (Eigen::VectorXd(12) <<
    //  20,20,20,
    //  520,520,525,
    //  2,2,2,
    //  5,5,10).finished().asDiagonal();

    R.setZero();
    R.diagonal() << 1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8;

    P = 10.0 * Q;
  }
};

// ===============================
// 期望轨迹状态容器 - 单个时刻的期望状态
// 格式遵循 Jacobian.cpp 约定：13维向量
// [qw qx qy qz px py pz wx wy wz vx vy vz]
//  |四元数姿态| |位置| |角速度| |线速度|
// 用途：从控制器传入当前时刻的期望状态给GMPC
// ===============================
struct DesiredState13 {
  Eigen::Matrix<double, 13, 1> v = Eigen::Matrix<double, 13, 1>::Zero();
};

// ===============================
// Pimpl（指针实现）设计模式说明
// ===============================
// 
public:
  GMPCDualLayer();
  ~GMPCDualLayer();

  // 重置求解器状态
  void reset();
  
  // 设置参数
  void setParams(const GMPCParams& p);
  
  // 设置时间步长（避免每次都setParams）
  void setDt(double dt);

  // 主计算接口：从Franka状态计算控制力矩
  // 返回值：成功true，失败false
  // tau_cmd：输出的7维控制力矩（不含重力补偿）
  bool computeTauMPC(
      const Eigen::Affine3d& O_T_EE,
      const Eigen::Matrix<double,7,1>& q,
      const Eigen::Matrix<double,7,1>& dq,
      const Eigen::Matrix<double,6,7>& J,
      const Eigen::Matrix<double,6,7>& Jdot,
      const Eigen::Matrix<double,7,7>& M,
      const Eigen::Matrix<double,7,1>& C,
      const Eigen::Matrix<double,7,1>& G,
      const DesiredState13& xd0,
      Eigen::Matrix<double,7,1>* tau_cmd);

private:
  // Pimpl模式：隐藏实现细节
  std::unique_ptr<DualLayerGMPC> impl_;
};

} // namespace serl_franka_controllers
