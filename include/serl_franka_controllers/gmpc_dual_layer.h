#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <OsqpEigen/OsqpEigen.h>
#include <array>
#include <vector>
#include <cstdint>

namespace serl_franka_controllers {

// ===============================
// Parameters (来自你 7dof-双层.txt + Jacobian.cpp 的风格)
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
  double delta_deviation = 0.001;   // 你文件里默认 0.001
  double lambda_dls      = 0.01;    // DLS base

  // performance tolerance (alpha_tolerance) — 你 MATLAB 有，但这里先留接口
  double alpha_tolerance = 0.95;

  GMPCParams() {
    umax << 50, 50, 50, 28, 28, 28, 28;
    umin = -umax;

    // Jacobian.cpp: Q_ / R_ 的量级，我按你文件风格给默认（你要自己调）
    Q.setZero();
    Q.block<3,3>(0,0) = 20.0 * Eigen::Matrix3d::Identity();     // rotation err
    Q.block<3,3>(3,3) = 20.0 * Eigen::Matrix3d::Identity();     // translation err
    Q.block<6,6>(6,6) = 800.0 * Eigen::Matrix<double,6,6>::Identity(); // twist (or vel) err part

    R.setZero();
    R.diagonal() << 1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8;

    P = 10.0 * Q;
  }
};

// ===============================
// Minimal "desired trajectory" container
// We follow Jacobian.cpp convention:
// xd row: [qw qx qy qz px py pz wx wy wz vx vy vz]  => 13 dims
// ===============================
struct DesiredState13 {
  Eigen::Matrix<double, 13, 1> v = Eigen::Matrix<double, 13, 1>::Zero();
};

// ===============================
// Dual-layer GMPC solver (OSQP warm-start + update)
// ===============================
class GMPCDualLayer {
public:
  GMPCDualLayer();

  void reset();
  void setParams(const GMPCParams& p);

  // Call once after you know dt/Nt etc.
  bool initSolver();

  // Main pure-function style API:
  // inputs:
  //   - current EE pose/orientation (from Franka state)
  //   - current q,dq, Jacobian(6x7), coriolis, mass (7x7), gravity (7)
  //   - desired state (pose + desired twist/vel)
  // outputs:
  //   - tau_cmd (7)
  bool computeTauMPC(
      const Eigen::Affine3d& O_T_EE,
      const Eigen::Matrix<double,7,1>& q,
      const Eigen::Matrix<double,7,1>& dq,
      const Eigen::Matrix<double,6,7>& J,
      const Eigen::Matrix<double,6,7>& Jdot,     // discrete diff
      const Eigen::Matrix<double,7,7>& M,
      const Eigen::Matrix<double,7,1>& C,        // coriolis
      const Eigen::Matrix<double,7,1>& G,        // gravity
      const DesiredState13& xd0,
      Eigen::Matrix<double,7,1>* tau_cmd);

private:
  // ========== from Jacobian.cpp ==========
  static Eigen::Matrix3d skew(const Eigen::Vector3d& v);
  static Eigen::Matrix<double,6,6> calculateAdjoint(const Eigen::Matrix<double,6,1>& x);
  static Eigen::Matrix4d matrix_logarithm(const Eigen::Matrix4d& X);

  static Eigen::Matrix<double,12,1> calculateErrorVector(
      const DesiredState13& xd,
      const Eigen::Quaterniond& q0,
      const Eigen::Vector3d& p0,
      const Eigen::Matrix<double,6,1>& V0);

  // DARE (idare-like) — complete iterative solver
  static Eigen::MatrixXd solveDARE(
      const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
      const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
      int max_iter, double tol);

  // DLS pseudoinverse for nullspace projector
  static Eigen::Matrix<double,7,6> dampedLeastSquaresJinv(
      const Eigen::Matrix<double,6,7>& J, double lambda);

  // ========== QP assembly (from Jacobian.cpp structure) ==========
  void updateCostFunction(
      const Eigen::Matrix<double,12,1>& p0,
      const std::vector<DesiredState13>& xd_seq);

  void updateConstraints(
      const Eigen::Matrix<double,12,1>& p0,
      const std::vector<DesiredState13>& xd_seq,
      const Eigen::Matrix<double,7,7>& M,
      const Eigen::Matrix<double,7,7>& Cmat,          // joint-space coriolis matrix (approx)
      const Eigen::Matrix<double,7,1>& G,
      const Eigen::Matrix<double,6,7>& J,
      const Eigen::Matrix<double,6,7>& Jdot);

  // Solve upper QP => returns full solution (X,U,extra) + u_primary (first step)
  bool solveUpperQP(Eigen::VectorXd* full_solution, Eigen::Matrix<double,7,1>* u_primary);

  // Solve secondary QP (your MATLAB solveSecondaryMPC_Modified) => u_final (first step)
  bool solveSecondaryQP(
      const Eigen::VectorXd& upper_full_solution,
      const Eigen::Matrix<double,7,1>& u_primary_first,
      const Eigen::Matrix<double,7,1>& u_prev,
      const Eigen::Matrix<double,6,7>& J,
      Eigen::Matrix<double,7,1>* u_final_first);

private:
  GMPCParams p_;

  // OSQP upper solver state (same spirit as Jacobian.cpp)
  bool solver_initialized_ = false;
  OsqpEigen::Solver solver_upper_;

  Eigen::SparseMatrix<double> H_;
  Eigen::VectorXd q_;
  Eigen::SparseMatrix<double> A_;
  Eigen::VectorXd l_;
  Eigen::VectorXd u_;

  // warm start memory
  Eigen::VectorXd primal_last_;
  Eigen::VectorXd dual_last_;

  // secondary solver (separate OSQP)
  OsqpEigen::Solver solver_lower_;
  bool lower_initialized_ = false;

  Eigen::Matrix<double,7,1> u_prev_cycle_ = Eigen::Matrix<double,7,1>::Zero();
};

} // namespace serl_franka_controllers
