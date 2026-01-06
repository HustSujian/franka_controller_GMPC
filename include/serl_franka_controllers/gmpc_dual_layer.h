// ====================================================================================
// gmpc_dual_layer.h  (CPP兼容版)
// 双层 GMPC 控制器 —— 公共接口头文件（Pimpl）
//
// 关键：GMPCParams 保持与 gmpc_dual_layer.cpp 内部实现一致：
// - static constexpr Nx/Nu
// - 含 use_R_delta / R_delta / R_cross
// - 含 xmin_rot/xmax_rot/xmin_pos/xmax_pos/xmin_vel/xmax_vel
// - 含 du_max / du_cross_max
//
// 注意：不要在 .cpp 里再次定义 GMPCParams（会重复定义）
// ====================================================================================

#pragma once

#include <Eigen/Dense>
#include <memory>

namespace serl_franka_controllers {


// ===============================
// GMPCParams：可调参数（与 .cpp 兼容）
// ===============================
struct GMPCParams
{
  // 维度（固定）
  static constexpr int Nx = 12; // state: [phi(6); V(6)]
  static constexpr int Nu = 7;  // input: joint torque
  int Nt = 20;                  // horizon
  double dt = 0.005;            // sampling time

  // ------------------------------
  // 主层代价权重
  // ------------------------------
  Eigen::Matrix<double, Nx, Nx> Q = Eigen::Matrix<double, Nx, Nx>::Identity();
  Eigen::Matrix<double, Nx, Nx> P = Eigen::Matrix<double, Nx, Nx>::Identity();
  Eigen::Matrix<double, Nu, Nu> R = Eigen::Matrix<double, Nu, Nu>::Identity();

  // 可选：Δu 权重
  bool use_R_delta = false;
  Eigen::Matrix<double, Nu, Nu> R_delta = Eigen::Matrix<double, Nu, Nu>::Zero();
  Eigen::Matrix<double, Nu, Nu> R_cross = Eigen::Matrix<double, Nu, Nu>::Zero();

  // ------------------------------
  // 控制输入约束
  // ------------------------------
  Eigen::Matrix<double, Nu, 1> umin = (-50.0) * Eigen::Matrix<double, Nu, 1>::Ones();
  Eigen::Matrix<double, Nu, 1> umax = ( 50.0) * Eigen::Matrix<double, Nu, 1>::Ones();

  // 时域内力矩变化约束 |u_k - u_{k-1}| <= du_max
  Eigen::Matrix<double, Nu, 1> du_max = (20.0) * Eigen::Matrix<double, Nu, 1>::Ones();

  // 跨周期首步变化约束 |u_0 - u_prev| <= du_cross_max
  Eigen::Matrix<double, Nu, 1> du_cross_max = (10.0) * Eigen::Matrix<double, Nu, 1>::Ones();

  // ------------------------------
  // 状态（误差）约束（总约束）
  // ------------------------------
  Eigen::Matrix<double, Nx, 1> xmin = (-10.0) * Eigen::Matrix<double, Nx, 1>::Ones();
  Eigen::Matrix<double, Nx, 1> xmax = ( 10.0) * Eigen::Matrix<double, Nx, 1>::Ones();

  // 可选：分块约束
  Eigen::Matrix<double, 3, 1> xmin_rot = (-10.0) * Eigen::Matrix<double, 3, 1>::Ones();
  Eigen::Matrix<double, 3, 1> xmax_rot = ( 10.0) * Eigen::Matrix<double, 3, 1>::Ones();

  Eigen::Matrix<double, 3, 1> xmin_pos = (-0.2) * Eigen::Matrix<double, 3, 1>::Ones();
  Eigen::Matrix<double, 3, 1> xmax_pos = ( 0.2) * Eigen::Matrix<double, 3, 1>::Ones();

  Eigen::Matrix<double, 6, 1> xmin_vel = (-2.0) * Eigen::Matrix<double, 6, 1>::Ones();
  Eigen::Matrix<double, 6, 1> xmax_vel = ( 2.0) * Eigen::Matrix<double, 6, 1>::Ones();

  // ------------------------------
  // 副层参数
  // ------------------------------
  double alpha_tolerance  = 0.95;
  double delta_deviation  = 0.001; // feasible box around primary solution
  Eigen::Matrix<double, Nu, Nu> R_null = Eigen::Matrix<double, Nu, Nu>::Identity();
  double w_smooth = 1e-8;
  double w_null   = 1e-6;

  GMPCParams()
  {
    
    Q.setZero();
    Q.block<3,3>(0,0) = 20.0 * Eigen::Matrix3d::Identity();
    Q.block<3,3>(3,3) = 1500.0 * Eigen::Matrix3d::Identity();
    Q.block<6,6>(6,6) = 20.0 * Eigen::Matrix<double,6,6>::Identity();

    R.setZero();
    R.diagonal().setConstant(1e-6);

    P = 10.0 * Q;

    // null 权重（按你之前那组）
    R_null.setZero();
    // R_null.diagonal() << 0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1;

    // Franka 关节力矩安全范围
    umax << 87, 87, 87, 87, 12, 12, 12;
    umin = -umax;
  }
};

// ===============================
// DesiredState13：单个时刻的期望状态容器
// [qw qx qy qz px py pz wx wy wz vx vy vz]
// ===============================
struct DesiredState13 {
  Eigen::Matrix<double, 13, 1> v = Eigen::Matrix<double, 13, 1>::Zero();
};

    // Trajectory helper (used by controllers)
void buildSpiralDesiredState13(double t, DesiredState13* xd0);

// Pimpl forward decl
class DualLayerGMPC;

// ===============================
// GMPCDualLayer：公共接口类
// ===============================
class GMPCDualLayer {
public:
  GMPCDualLayer();
  ~GMPCDualLayer();

  GMPCDualLayer(const GMPCDualLayer&) = delete;
  GMPCDualLayer& operator=(const GMPCDualLayer&) = delete;

  GMPCDualLayer(GMPCDualLayer&&) noexcept;
  GMPCDualLayer& operator=(GMPCDualLayer&&) noexcept;

  void reset();
  void setParams(const GMPCParams& p);
  void setDt(double dt);

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
  std::unique_ptr<DualLayerGMPC> impl_;
};

} // namespace serl_franka_controllers
