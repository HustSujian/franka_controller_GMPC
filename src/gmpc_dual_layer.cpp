// gmpc_dual_layer.cpp
// 7 自由度力矩控制的双层 GMPC（主层跟踪 + 副层零空间/平滑度优化）
// - 主层 QP：在误差状态 y=[phi; V] 上做跟踪并施加约束
// - 副层 QP：在主解附近的偏差盒中优化平滑度和零空间偏好
//
// 此实现参考了上传的 MATLAB 文件 “7dof-双层.txt”：
//   - solvePrimaryMPC() + buildPrimaryConstraints() + buildPrimaryCost()
//   - solveSecondaryMPC_Modified() + 偏差约束 + 平滑项 + N'RnullN
//
// 同时遵循了 Jacobian.cpp 中的 OsqpEigen 用法。
//
// 注（Franka 实际应用）：
// - Franka 通过 model handle 提供质量矩阵 M(q)、科里奥利向量 c(q,dq) 与重力 g(q)。
// - 不提供科里奥利矩阵 C(q,dq)，Jdot 也无法直接获取。
// - 因此采用仿射的 Vdot 模型：
//     Vdot = J * M^{-1} * u  +  (Jdot*dq - J*M^{-1}(coriolis+gravity))
//   并用有限差分估计 Jdot。

#include <OsqpEigen/OsqpEigen.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <unsupported/Eigen/MatrixFunctions>

namespace serl_franka_controllers {

// ------------------------------ 工具函数 ------------------------------

static inline Eigen::Matrix3d skew3(const Eigen::Vector3d& a) {
  Eigen::Matrix3d A;
  A << 0.0, -a.z(), a.y(),
       a.z(), 0.0, -a.x(),
      -a.y(), a.x(), 0.0;
  return A;
}
// 666
// 李代数中 twist V = [w; v]（R^6）的伴随算子 ad_V
// ad_V 的定义： [ [w^, 0],
//                [v^, w^] ]
static inline Eigen::Matrix<double, 6, 6> ad6(const Eigen::Matrix<double, 6, 1>& V) {
  const Eigen::Vector3d w = V.template head<3>();
  const Eigen::Vector3d v = V.template tail<3>();
  Eigen::Matrix<double, 6, 6> ad;
  ad.setZero();
  ad.block<3,3>(0,0) = skew3(w);
  ad.block<3,3>(3,0) = skew3(v);
  ad.block<3,3>(3,3) = skew3(w);
  return ad;
}

// J (6x7) 的阻尼伪逆：J^T (J J^T + lambda^2 I)^-1
static inline Eigen::Matrix<double, 7, 6> dampedPinvJT(
    const Eigen::Matrix<double, 6, 7>& J, double lambda) {
  Eigen::Matrix<double, 6, 6> JJt = J * J.transpose();
  Eigen::Matrix<double, 6, 6> reg = JJt + (lambda * lambda) * Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 7, 6> Jinv = J.transpose() * reg.inverse();
  return Jinv;
}

// 利用 SVD 近似计算 J 的条件数
static inline double condNumber(const Eigen::Matrix<double, 6, 7>& J) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto& s = svd.singularValues();
  if (s.size() == 0) return std::numeric_limits<double>::infinity();
  const double smax = s(0);
  const double smin = s(s.size() - 1);
  if (smin <= 1e-12) return std::numeric_limits<double>::infinity();
  return smax / smin;
}

static inline double manipulabilityMeasure(const Eigen::Matrix<double, 6, 7>& J) {
  // sqrt(det(J J^T)) 可用 SVD 的奇异值乘积计算
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto& s = svd.singularValues();
  double prod = 1.0;
  for (int i = 0; i < s.size(); ++i) prod *= s(i);
  return prod;
}

// SE(3) 对数：输入 T ∈ SE(3)，输出 se(3) 的 4x4 矩阵
// 这里直接用 Eigen 的矩阵对数（对一般矩阵亦可），再按 MATLAB 取 phi：
//   w = [X0(3,2); X0(1,3); X0(2,1)]   （1 基索引）
//   p = X0(1:3,4)
static inline Eigen::Matrix<double, 6, 1> se3LogPhi(const Eigen::Matrix4d& T_err) {
  Eigen::Matrix4d X0 = T_err.log();  // 依赖 MatrixFunctions 模块
  Eigen::Matrix<double, 6, 1> phi;
  phi(0) = X0(2,1);  // MATLAB 中的 (3,2)
  phi(1) = X0(0,2);  // MATLAB 中的 (1,3)
  phi(2) = X0(1,0);  // MATLAB 中的 (2,1)
  phi(3) = X0(0,3);
  phi(4) = X0(1,3);
  phi(5) = X0(2,3);
  return phi;
}

static inline Eigen::Matrix4d poseToT(const Eigen::Quaterniond& q, const Eigen::Vector3d& p) {
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3,3>(0,0) = q.normalized().toRotationMatrix();
  T.block<3,1>(0,3) = p;
  return T;
}

// 稠密矩阵转稀疏矩阵（行优先构造）
static inline Eigen::SparseMatrix<double> denseToSparse(const Eigen::MatrixXd& D, double prune_eps = 0.0) {
  Eigen::SparseMatrix<double> S(D.rows(), D.cols());
  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(static_cast<size_t>(D.rows() * D.cols()));
  for (int r = 0; r < D.rows(); ++r) {
    for (int c = 0; c < D.cols(); ++c) {
      const double v = D(r,c);
      if (std::abs(v) > prune_eps) trips.emplace_back(r, c, v);
    }
  }
  S.setFromTriplets(trips.begin(), trips.end());
  return S;
}

static inline void addBlockTriplets(std::vector<Eigen::Triplet<double>>& T,
                                    int row0, int col0,
                                    const Eigen::MatrixXd& B,
                                    double prune_eps = 0.0) {
  for (int r = 0; r < B.rows(); ++r) {
    for (int c = 0; c < B.cols(); ++c) {
      const double v = B(r,c);
      if (std::abs(v) > prune_eps) T.emplace_back(row0 + r, col0 + c, v);
    }
  }
}

static inline void addIdentityTriplets(std::vector<Eigen::Triplet<double>>& T,
                                       int row0, int col0, int n, double scale = 1.0) {
  for (int i = 0; i < n; ++i) {
    T.emplace_back(row0 + i, col0 + i, scale);
  }
}

// ------------------------------ 参数定义 ------------------------------

struct GMPCParams {
  // 维度（本实现固定）
  static constexpr int Nx = 12;  // 状态: [phi(6); V(6)]
  static constexpr int Nu = 7;   // 控制: 关节力矩（或力矩控制项）
  int Nt = 10;                   // 预测时域长度
  double dt = 0.001;             // 控制周期，应与 Franka 回路一致

  // 主层代价权重
  Eigen::Matrix<double, 12, 12> Q = Eigen::Matrix<double, 12, 12>::Identity();
  Eigen::Matrix<double, 12, 12> P = Eigen::Matrix<double, 12, 12>::Identity();
  Eigen::Matrix<double, 7, 7>  R = 1e-8 * Eigen::Matrix<double, 7, 7>::Identity();

  // 可选的主层 Δu 权重（类似 MATLAB 的 param.R_delta）
  bool use_R_delta = false;
  Eigen::Matrix<double, 7, 7> R_delta = Eigen::Matrix<double, 7, 7>::Zero();
  Eigen::Matrix<double, 7, 7> R_cross = Eigen::Matrix<double, 7, 7>::Zero();

  // 控制输入约束
  Eigen::Matrix<double, 7, 1> umin = (-50.0) * Eigen::Matrix<double, 7, 1>::Ones();
  Eigen::Matrix<double, 7, 1> umax = ( 50.0) * Eigen::Matrix<double, 7, 1>::Ones();

  // 时域内的力矩变化硬约束 (|u_k - u_{k-1}| <= max)
  Eigen::Matrix<double, 7, 1> du_max = (20.0) * Eigen::Matrix<double, 7, 1>::Ones();

  // 跨周期首步的力矩变化约束（围绕上一周期 u_prev）
  Eigen::Matrix<double, 7, 1> du_cross_max = (10.0) * Eigen::Matrix<double, 7, 1>::Ones();

  // 状态（误差）约束，风格参考 MATLAB，具体数值可按应用调节
  Eigen::Matrix<double, 12, 1> xmin = (-10.0) * Eigen::Matrix<double, 12, 1>::Ones();
  Eigen::Matrix<double, 12, 1> xmax = ( 10.0) * Eigen::Matrix<double, 12, 1>::Ones();

  // 相对于目标的实际位姿误差约束（位置/姿态/速度）
  Eigen::Matrix<double, 3, 1> xmin_rot = (-10.0) * Eigen::Matrix<double, 3, 1>::Ones();
  Eigen::Matrix<double, 3, 1> xmax_rot = ( 10.0) * Eigen::Matrix<double, 3, 1>::Ones();

  Eigen::Matrix<double, 3, 1> xmin_pos = (-0.2) * Eigen::Matrix<double, 3, 1>::Ones();
  Eigen::Matrix<double, 3, 1> xmax_pos = ( 0.2) * Eigen::Matrix<double, 3, 1>::Ones();

  Eigen::Matrix<double, 6, 1> xmin_vel = (-2.0) * Eigen::Matrix<double, 6, 1>::Ones();
  Eigen::Matrix<double, 6, 1> xmax_vel = ( 2.0) * Eigen::Matrix<double, 6, 1>::Ones();

  // 副层参数
  double alpha_tolerance = 0.95;
  double delta_deviation = 0.001;  // 主层解周围的偏差盒大小
  Eigen::Matrix<double, 7, 7> R_null = Eigen::Matrix<double, 7, 7>::Identity();
  double w_smooth = 1e-8;
  double w_null = 1e-6;
};

// ------------------------------ 输入结构体 ------------------------------

struct GMPCInput {
  // 当前机器人状态
  Eigen::Matrix<double, 7, 1> q;
  Eigen::Matrix<double, 7, 1> dq;

  // 基座坐标系（Franka 的 O 框架）下的末端位姿
  Eigen::Quaterniond orientation;
  Eigen::Vector3d position;

  // 期望末端位姿
  Eigen::Quaterniond orientation_d;
  Eigen::Vector3d position_d;

  // 基座系的期望 twist Vd（6x1）
  Eigen::Matrix<double, 6, 1> Vd;

  // 模型项
  Eigen::Matrix<double, 7, 7> M;
  Eigen::Matrix<double, 7, 1> coriolis;
  Eigen::Matrix<double, 7, 1> gravity;

  // 基座系雅可比及可选的 Jdot 估计
  Eigen::Matrix<double, 6, 7> J;
};

// ------------------------------ 双层 GMPC 求解器 ------------------------------

class DualLayerGMPC {
public:
  DualLayerGMPC() = default;

  void setParams(const GMPCParams& p) {
    params_ = p;
    resetWarmStart();
  }

  const GMPCParams& params() const { return params_; }

  void resetWarmStart() {
    std::lock_guard<std::mutex> lock(mutex_);
    primary_initialized_ = false;
    secondary_initialized_ = false;
    primary_primal_.resize(0);
    primary_dual_.resize(0);
    secondary_primal_.resize(0);
    secondary_dual_.resize(0);
    J_prev_valid_ = false;
    J_prev_.setZero();
    last_u_cmd_.setZero();
    last_u_prev_valid_ = false;
  }

  // 主入口：计算力矩控制项 u（7x1）。
  // 在 CartesianImpedanceController 中的推荐用法：
  //   u = gmpc.computeTau(in);
  //   tau_cmd = u + coriolis; （随后再做力矩变化率饱和）
  Eigen::Matrix<double, 7, 1> computeTau(const GMPCInput& in) {
    std::lock_guard<std::mutex> lock(mutex_);

    // ----------- 构造 p0（初始状态） = [phi; V_current] -----------
    // MATLAB: X00 = X00^{-1} * Xd; X0 = logm(X00); phi 来自 X0，再拼上当前 twist x0(8:end)。
    // 此处：phi = log( T_cur^{-1} T_des )，V_current = J * dq
    const Eigen::Matrix4d T_cur = poseToT(in.orientation, in.position);
    const Eigen::Matrix4d T_des = poseToT(in.orientation_d, in.position_d);
    const Eigen::Matrix4d T_err = T_cur.inverse() * T_des;

    const Eigen::Matrix<double, 6, 1> phi = se3LogPhi(T_err);
    const Eigen::Matrix<double, 6, 1> V_cur = in.J * in.dq;

    Eigen::Matrix<double, 12, 1> p0;
    p0 << phi, V_cur;

    // 预测时域的期望 twist 轨迹：沿用 MATLAB xid(k,:) = 期望 twist 样本。
    // 这里简化为时域内常值，可替换为外部提供的序列。
    std::vector<Eigen::Matrix<double, 6, 1>> Vd_seq(params_.Nt);
    for (int k = 0; k < params_.Nt; ++k) Vd_seq[k] = in.Vd;

    // ----------- 自适应阻尼与零空间投影 N -----------
    const double condJ = condNumber(in.J);
    const double manip = manipulabilityMeasure(in.J);

    double lambda_base = 0.01;
    double lambda;
    if (condJ > 10.0) {
      lambda = lambda_base * condJ / 10.0;
    } else if (manip < 0.01) {
      lambda = lambda_base * (0.01 / std::max(manip, 1e-12));
    } else {
      lambda = lambda_base;
    }
    lambda = std::min(lambda, 0.2);

    const Eigen::Matrix<double, 7, 6> J_pinv = dampedPinvJT(in.J, lambda);
    const Eigen::Matrix<double, 7, 7> Nproj = Eigen::Matrix<double, 7, 7>::Identity() - J_pinv * in.J;

    // ----------- 主层求解 -----------
    Eigen::VectorXd sol1_x;
    int status1 = 0;
    Eigen::Matrix<double, 7, 1> u_primary = Eigen::Matrix<double, 7, 1>::Zero();

    {
      PrimaryQP qp1 = buildPrimaryQP(in, p0, Vd_seq);
      status1 = solveOSQPPrimary(qp1, &sol1_x);
      if (status1 == 1 && sol1_x.size() == qp1.nvar) {
        // 取 u_primary = k=0 时刻的第一个控制量
        const int state_vars = (params_.Nt + 1) * GMPCParams::Nx;
        u_primary = sol1_x.segment(state_vars, GMPCParams::Nu);
      } else {
        // 回退：主层失败则输出零控制
        u_primary.setZero();
        // 仍然记录 last_u_cmd_ 以保持连续性
        last_u_cmd_ = u_primary;
        last_u_prev_valid_ = true;
        return u_primary;
      }
    }

    // ----------- 构建副层可行集 -----------
    FeasibleSet fs;
    fs.full_solution = sol1_x;
    fs.u_primary_first = u_primary;
    fs.alpha = params_.alpha_tolerance;
    fs.deviation_bound = params_.delta_deviation;

    // ----------- 副层求解 -----------
    Eigen::Matrix<double, 7, 1> u_final = u_primary;
    int status2 = 0;
    Eigen::VectorXd sol2_x;

    {
      SecondaryQP qp2 = buildSecondaryQP(fs, Nproj);
      status2 = solveOSQPSecondary(qp2, &sol2_x);

      if (status2 == 1 && sol2_x.size() == qp2.nvar) {
        // sol2 形如 [u0; u1; ... u_{Nt-1}]（Nu*Nt）
        u_final = sol2_x.segment(0, GMPCParams::Nu);
        // 性能下降检查（同 MATLAB 思路）
        const double degr = evaluatePerformanceDegradation(u_final, u_primary);
        if (degr > (1.0 - fs.alpha)) {
          // 下降过大则保留主层解
          u_final = u_primary;
        }
      } else {
        u_final = u_primary;
      }
    }

    last_u_cmd_ = u_final;
    last_u_prev_valid_ = true;
    return u_final;
  }

  // 提供上一周期控制量，用于跨周期约束
  bool hasLastU() const { return last_u_prev_valid_; }
  Eigen::Matrix<double, 7, 1> lastU() const { return last_u_cmd_; }

private:
  // ---------------- 主层 QP 容器 ----------------
  struct PrimaryQP {
    int nvar = 0;
    int ncon = 0;
    Eigen::SparseMatrix<double> H;
    Eigen::VectorXd g;
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
  };

  // ---------------- 副层 QP 容器 ----------------
  struct SecondaryQP {
    int nvar = 0;
    int ncon = 0;
    Eigen::SparseMatrix<double> H;
    Eigen::VectorXd g;
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
  };

  struct FeasibleSet {
    Eigen::VectorXd full_solution;          // 主层完整解
    Eigen::Matrix<double, 7, 1> u_primary_first;
    double alpha = 0.95;
    double deviation_bound = 0.001;
  };

  // 估计 Jdot
  Eigen::Matrix<double, 6, 7> estimateJdot(const Eigen::Matrix<double, 6, 7>& J, double dt) {
    Eigen::Matrix<double, 6, 7> Jdot = Eigen::Matrix<double, 6, 7>::Zero();
    if (J_prev_valid_) {
      Jdot = (J - J_prev_) / std::max(dt, 1e-6);
    }
    J_prev_ = J;
    J_prev_valid_ = true;
    return Jdot;
  }

  // ---------------- 构建主层 QP ----------------
  PrimaryQP buildPrimaryQP(const GMPCInput& in,
                           const Eigen::Matrix<double, 12, 1>& p0,
                           const std::vector<Eigen::Matrix<double, 6, 1>>& Vd_seq) {
    const int Nx = GMPCParams::Nx;
    const int Nu = GMPCParams::Nu;
    const int Nt = params_.Nt;
    const double dt = params_.dt;

    // 变量：X(0..Nt) (Nx*(Nt+1)) + U(0..Nt-1) (Nu*Nt)
    const int nvar = Nx * (Nt + 1) + Nu * Nt;

    // 约束块：
    // 1) 动力学：Nx*Nt
    // 2) 初始状态：Nx
    // 3) 状态约束 k=1..Nt：Nx*Nt
    // 4) 控制约束：Nu*Nt
    // 5) 跨周期首步变化：Nu（若存在 last_u）
    // 6) 时域内 Δu 硬约束：Nu*(Nt-1)
    const bool has_prev = last_u_prev_valid_;
    const int ncon =
        (Nx * Nt) + Nx + (Nx * Nt) + (Nu * Nt) + (has_prev ? Nu : 0) + (Nu * (Nt - 1));

    PrimaryQP qp;
    qp.nvar = nvar;
    qp.ncon = ncon;

    // --------- 组装 H 与 g（代价） ---------
    // 对应 MATLAB buildPrimaryCost():
    // 对 k=1..Nt:
    //   C = I; C(7:12,1:6) = -ad(Vd_k)
    //   b = [0; Vd_k] 用于线性项 q: -C'Q b
    //   状态块: C'Q C（末尾用 C'P C）
    // 控制块: R（可选 Δu）
    //
    // 使用三元组构建稀疏矩阵。
    std::vector<Eigen::Triplet<double>> Ht;
    Ht.reserve(static_cast<size_t>(nvar * 5));  // 粗略预估容量

    qp.g = Eigen::VectorXd::Zero(nvar);

    for (int k = 1; k <= Nt; ++k) {
      Eigen::Matrix<double, 12, 12> Cmap = Eigen::Matrix<double, 12, 12>::Identity();
      Cmap.block<6,6>(6,0) = -ad6(Vd_seq[std::min(k-1, Nt-1)]);

      Eigen::Matrix<double, 12, 12> W = (k < Nt) ? params_.Q : params_.P;
      Eigen::Matrix<double, 12, 12> Hblk = Cmap.transpose() * W * Cmap;

      Eigen::Matrix<double, 12, 1> bvec = Eigen::Matrix<double, 12, 1>::Zero();
      bvec.segment<6>(6) = Vd_seq[std::min(k-1, Nt-1)];
      Eigen::Matrix<double, 12, 1> gblk = -Cmap.transpose() * W * bvec;

      const int idx0 = k * Nx;  // X_k 起始索引 k*Nx
      // H 块
      for (int r = 0; r < Nx; ++r) {
        for (int c = 0; c < Nx; ++c) {
          const double v = Hblk(r,c);
          if (std::abs(v) > 0.0) Ht.emplace_back(idx0 + r, idx0 + c, v);
        }
      }
      // g 块
      qp.g.segment(idx0, Nx) += gblk;
    }

    // 控制代价：U 序列上块对角的 R
    const int u_offset = Nx * (Nt + 1);
    for (int k = 0; k < Nt; ++k) {
      const int uk = u_offset + k * Nu;
      for (int r = 0; r < Nu; ++r) {
        for (int c = 0; c < Nu; ++c) {
          const double v = params_.R(r,c);
          if (std::abs(v) > 0.0) Ht.emplace_back(uk + r, uk + c, v);
        }
      }
    }

    // 可选 Δu 代价（与 MATLAB 一致）
    if (params_.use_R_delta) {
      for (int k = 0; k < Nt; ++k) {
        const int uk = u_offset + k * Nu;
        if (k == 0) {
          if (has_prev) {
            // (u0 - u_prev)^T R_cross (u0 - u_prev)
            for (int r = 0; r < Nu; ++r) {
              for (int c = 0; c < Nu; ++c) {
                const double v = params_.R_cross(r,c);
                if (std::abs(v) > 0.0) Ht.emplace_back(uk + r, uk + c, v);
              }
            }
            qp.g.segment(uk, Nu) += -(params_.R_cross * last_u_cmd_);
          }
        } else {
          // (uk - u_{k-1})^T R_delta (uk - u_{k-1})
          const int ukm1 = u_offset + (k - 1) * Nu;
          // uk^T R uk 项
          for (int r = 0; r < Nu; ++r) {
            for (int c = 0; c < Nu; ++c) {
              const double v = params_.R_delta(r,c);
              if (std::abs(v) > 0.0) Ht.emplace_back(uk + r, uk + c, v);
            }
          }
          // u_{k-1}^T R u_{k-1} 项
          for (int r = 0; r < Nu; ++r) {
            for (int c = 0; c < Nu; ++c) {
              const double v = params_.R_delta(r,c);
              if (std::abs(v) > 0.0) Ht.emplace_back(ukm1 + r, ukm1 + c, v);
            }
          }
          // 交叉项 -R
          for (int r = 0; r < Nu; ++r) {
            for (int c = 0; c < Nu; ++c) {
              const double v = -params_.R_delta(r,c);
              if (std::abs(v) > 0.0) {
                Ht.emplace_back(uk + r, ukm1 + c, v);
                Ht.emplace_back(ukm1 + r, uk + c, v);
              }
            }
          }
        }
      }
    }

    // 对角正则化以保证半正定
    for (int i = 0; i < nvar; ++i) {
      Ht.emplace_back(i, i, 1e-9);
    }

    qp.H.resize(nvar, nvar);
    qp.H.setFromTriplets(Ht.begin(), Ht.end());

    // --------- 组装约束 A, l, u ---------
    std::vector<Eigen::Triplet<double>> At;
    At.reserve(static_cast<size_t>(ncon * 10));
    qp.l = Eigen::VectorXd::Constant(ncon, -std::numeric_limits<double>::infinity());
    qp.u = Eigen::VectorXd::Constant(ncon,  std::numeric_limits<double>::infinity());

    int row = 0;

    // (1) 动力学约束：X_{k+1} = Ad X_k + Bd u_k + hd
    // MATLAB buildPrimaryConstraints 形式：
    //  Ac = [ -ad(Vd), -I;
    //         0,       H ]
    //  Bc = [ 0;
    //         F ]
    //  hc = [ Vd;
    //         b ]
    //
    // 这里使用实际可获得的仿射模型：
    //  phi_dot = -ad(Vd)*phi + V - Vd
    //  V_dot   = J*M^{-1}*u + (Jdot*dq - J*M^{-1}(coriolis+gravity))
    //
    // 因此：
    //  d/dt [phi] = (-ad(Vd))*phi + I*V + (-I)*Vd
    //  d/dt [V]   = 0*phi + 0*V + (J*M^{-1})*u + b_aff
    //
    // 离散化：
    //  X_{k+1} = (I + Ac dt) X_k + (Bc dt) u_k + (hc dt)
    //
    const Eigen::Matrix<double, 7, 7> Minv = in.M.inverse();
    const Eigen::Matrix<double, 6, 7> J = in.J;
    const Eigen::Matrix<double, 6, 7> Jdot = estimateJdot(in.J, dt);

    const Eigen::Matrix<double, 6, 7> F = J * Minv;  // 6x7

    // b_aff = Jdot*dq - J*M^{-1}*(coriolis+gravity)
    const Eigen::Matrix<double, 7, 1> tau_bias = in.coriolis + in.gravity;
    const Eigen::Matrix<double, 6, 1> b_aff = (Jdot * in.dq) - (J * (Minv * tau_bias));

    for (int k = 0; k < Nt; ++k) {
      const Eigen::Matrix<double, 6, 1> Vd = Vd_seq[k];

      Eigen::Matrix<double, 12, 12> Ac = Eigen::Matrix<double, 12, 12>::Zero();
      Ac.block<6,6>(0,0) = -ad6(Vd);
      Ac.block<6,6>(0,6) = -Eigen::Matrix<double,6,6>::Identity();
      // V 的动力学在此简化模型中与 phi、V 无耦合（如需可扩展）
      // Ac.block<6,6>(6,0) = 0;
      // Ac.block<6,6>(6,6) = 0;

      Eigen::Matrix<double, 12, 7> Bc = Eigen::Matrix<double, 12, 7>::Zero();
      Bc.block<6,7>(6,0) = F;

      Eigen::Matrix<double, 12, 1> hc = Eigen::Matrix<double, 12, 1>::Zero();
      hc.segment<6>(0) = Vd;
      hc.segment<6>(6) = b_aff;

      Eigen::Matrix<double, 12, 12> Ad = Eigen::Matrix<double, 12, 12>::Identity() + Ac * dt;
      Eigen::Matrix<double, 12, 7>  Bd = Bc * dt;
      Eigen::Matrix<double, 12, 1>  hd = hc * dt;

      // 约束形式：
      //  -Ad * X_k + I * X_{k+1} + (-Bd)*u_k = hd
      const int xk  = k * Nx;
      const int xk1 = (k + 1) * Nx;
      const int uk  = u_offset + k * Nu;

      // X_k 上的 -Ad
      addBlockTriplets(At, row, xk, -Ad, 0.0);
      // X_{k+1} 上的 +I
      addIdentityTriplets(At, row, xk1, Nx, 1.0);
      // u_k 上的 -Bd
      addBlockTriplets(At, row, uk, -Bd, 0.0);

      qp.l.segment(row, Nx) = hd;
      qp.u.segment(row, Nx) = hd;

      row += Nx;
    }

    // (2) 初始状态：X_0 = p0
    {
      addIdentityTriplets(At, row, 0, Nx, 1.0);
      qp.l.segment(row, Nx) = p0;
      qp.u.segment(row, Nx) = p0;
      row += Nx;
    }

    // (3) 状态约束 k=1..Nt
    // MATLAB 相对目标位姿的约束：
    //  rot -> phi(1:3), pos -> phi(4:6), vel -> V(1:6)
    for (int k = 1; k <= Nt; ++k) {
      const int xk = k * Nx;

      Eigen::Matrix<double, 12, 1> xmin = Eigen::Matrix<double, 12, 1>::Zero();
      Eigen::Matrix<double, 12, 1> xmax = Eigen::Matrix<double, 12, 1>::Zero();

      // phi 约束
      xmin.segment<3>(0) = params_.xmin_rot;
      xmax.segment<3>(0) = params_.xmax_rot;
      xmin.segment<3>(3) = params_.xmin_pos;
      xmax.segment<3>(3) = params_.xmax_pos;

      // V 约束
      xmin.segment<6>(6) = params_.xmin_vel;
      xmax.segment<6>(6) = params_.xmax_vel;

      // 约束：xmin <= X_k <= xmax
      addIdentityTriplets(At, row, xk, Nx, 1.0);
      qp.l.segment(row, Nx) = xmin;
      qp.u.segment(row, Nx) = xmax;
      row += Nx;
    }

    // (4) 控制约束 k=0..Nt-1：umin <= u_k <= umax
    for (int k = 0; k < Nt; ++k) {
      const int uk = u_offset + k * Nu;
      addIdentityTriplets(At, row, uk, Nu, 1.0);
      qp.l.segment(row, Nu) = params_.umin;
      qp.u.segment(row, Nu) = params_.umax;
      row += Nu;
    }

    // (5) 跨周期首步变化围绕 last_u_cmd_
    if (has_prev) {
      const int u0 = u_offset + 0 * Nu;
      addIdentityTriplets(At, row, u0, Nu, 1.0);
      qp.l.segment(row, Nu) = last_u_cmd_ - params_.du_cross_max;
      qp.u.segment(row, Nu) = last_u_cmd_ + params_.du_cross_max;
      row += Nu;
    }

    // (6) 时域内 Δu 约束：|u_k - u_{k-1}| <= du_max，k=1..Nt-1
    for (int k = 1; k < Nt; ++k) {
      const int uk = u_offset + k * Nu;
      const int ukm1 = u_offset + (k - 1) * Nu;

      // u_k - u_{k-1} <= du_max 且 >= -du_max
      // 通过 A * z ∈ [l,u] 实现，A = [uk 的 I, ukm1 的 -I]
      // l = -du_max, u = du_max
      for (int i = 0; i < Nu; ++i) {
        At.emplace_back(row + i, uk + i, 1.0);
        At.emplace_back(row + i, ukm1 + i, -1.0);
      }
      qp.l.segment(row, Nu) = -params_.du_max;
      qp.u.segment(row, Nu) =  params_.du_max;
      row += Nu;
    }

    if (row != ncon) {
      // 安全检查
      throw std::runtime_error("Primary QP constraint count mismatch: row != ncon");
    }

    qp.A.resize(ncon, nvar);
    qp.A.setFromTriplets(At.begin(), At.end());

    return qp;
  }

  // ---------------- 构建副层 QP ----------------
  SecondaryQP buildSecondaryQP(const FeasibleSet& fs,
                               const Eigen::Matrix<double, 7, 7>& Nproj) {
    const int Nu = GMPCParams::Nu;
    const int Nt = params_.Nt;

    // 变量：U_seq (Nu*Nt)
    const int nvar = Nu * Nt;

    // 约束：
    // (1) 主层 u_seq 周围的偏差盒：2*Nu*Nt
    // (2) 控制边界：2*Nu*Nt
    const int ncon = 4 * Nu * Nt;

    SecondaryQP qp;
    qp.nvar = nvar;
    qp.ncon = ncon;

    // 从完整解中提取主层的 u 序列：
    // 主层解排布为 [X(0..Nt); U(0..Nt-1)]
    const int state_vars = (params_.Nt + 1) * GMPCParams::Nx;
    const int control_vars = GMPCParams::Nu * params_.Nt;
    if (fs.full_solution.size() < state_vars + control_vars) {
      throw std::runtime_error("FeasibleSet full_solution has invalid size.");
    }
    Eigen::VectorXd u_primary_full = fs.full_solution.segment(state_vars, control_vars);

    // 将 u_primary_full 重塑为 Nu x Nt（MATLAB 里 reshape 为 Nt x Nu）
    std::vector<Eigen::Matrix<double, 7, 1>> uref_seq(params_.Nt);
    for (int k = 0; k < params_.Nt; ++k) {
      uref_seq[k] = u_primary_full.segment(k * Nu, Nu);
    }

    // --------- 副层代价：平滑 + 零空间偏好 ---------
    // 与 MATLAB 一致：M2 = sparse(total_vars,total_vars); q2 = zeros
    std::vector<Eigen::Triplet<double>> Ht;
    Ht.reserve(static_cast<size_t>(nvar * 8));
    qp.g = Eigen::VectorXd::Zero(nvar);

    // 平滑性：
    // k=0：若有上一周期则 (u0 - u_prev)^2，否则仅 u0^2
    // k>0： (uk - u_{k-1})^2
    const double ws = params_.w_smooth;
    if (ws > 0.0) {
      for (int k = 0; k < Nt; ++k) {
        const int uk = k * Nu;
        if (k == 0) {
          // 在 u0 上加入 ws * I
          for (int i = 0; i < Nu; ++i) Ht.emplace_back(uk + i, uk + i, ws);
          if (last_u_prev_valid_) {
            qp.g.segment(uk, Nu) += -(ws * last_u_cmd_);
          }
        } else {
          const int ukm1 = (k - 1) * Nu;
          // uk^2 项
          for (int i = 0; i < Nu; ++i) Ht.emplace_back(uk + i, uk + i, ws);
          // ukm1^2 项
          for (int i = 0; i < Nu; ++i) Ht.emplace_back(ukm1 + i, ukm1 + i, ws);
          // 交叉项 -ws
          for (int i = 0; i < Nu; ++i) {
            Ht.emplace_back(uk + i, ukm1 + i, -ws);
            Ht.emplace_back(ukm1 + i, uk + i, -ws);
          }
        }
      }
    }

    // 零空间偏好：sum_k u_k^T (N^T R_null N) u_k
    const double wn = params_.w_null;
    Eigen::Matrix<double, 7, 7> Nw = Nproj.transpose() * params_.R_null * Nproj;
    if (wn > 0.0) {
      for (int k = 0; k < Nt; ++k) {
        const int uk = k * Nu;
        for (int r = 0; r < Nu; ++r) {
          for (int c = 0; c < Nu; ++c) {
            const double v = wn * Nw(r,c);
            if (std::abs(v) > 0.0) Ht.emplace_back(uk + r, uk + c, v);
          }
        }
      }
    }

    // 正则化
    for (int i = 0; i < nvar; ++i) Ht.emplace_back(i, i, 1e-9);

    qp.H.resize(nvar, nvar);
    qp.H.setFromTriplets(Ht.begin(), Ht.end());

    // --------- 副层约束：偏差盒 + 边界 ---------
    std::vector<Eigen::Triplet<double>> At;
    At.reserve(static_cast<size_t>(ncon * 2));
    qp.l = Eigen::VectorXd::Constant(ncon, -std::numeric_limits<double>::infinity());
    qp.u = Eigen::VectorXd::Constant(ncon,  std::numeric_limits<double>::infinity());

    int row = 0;
    const double delta = fs.deviation_bound;

    // (1) 偏差约束：u <= uref + delta 且 u >= uref - delta
    // 拆成两组：
    //   +I*u ∈ [-inf, uref+delta]
    //   -I*u ∈ [-inf, -(uref-delta)]  <=> u >= uref-delta
    for (int k = 0; k < Nt; ++k) {
      const int uk = k * Nu;
      const Eigen::Matrix<double, 7, 1> uref = uref_seq[k];

      // 上界：u <= uref + delta
      for (int i = 0; i < Nu; ++i) At.emplace_back(row + i, uk + i, 1.0);
      qp.u.segment(row, Nu) = uref.array() + delta;
      row += Nu;

      // 下界：-u <= -(uref - delta)
      for (int i = 0; i < Nu; ++i) At.emplace_back(row + i, uk + i, -1.0);
      qp.u.segment(row, Nu) = -(uref.array() - delta);
      row += Nu;
    }

    // (2) 边界：umin <= u <= umax（双边）
    for (int k = 0; k < Nt; ++k) {
      const int uk = k * Nu;

      // 正向约束：+I*u ∈ [umin, umax]
      for (int i = 0; i < Nu; ++i) At.emplace_back(row + i, uk + i, 1.0);
      qp.l.segment(row, Nu) = params_.umin;
      qp.u.segment(row, Nu) = params_.umax;
      row += Nu;

      // 反向约束：-I*u ∈ [-umax, -umin]（冗余，但与 MATLAB 结构一致）
      for (int i = 0; i < Nu; ++i) At.emplace_back(row + i, uk + i, -1.0);
      qp.l.segment(row, Nu) = -params_.umax;
      qp.u.segment(row, Nu) = -params_.umin;
      row += Nu;
    }

    if (row != ncon) {
      throw std::runtime_error("Secondary QP constraint count mismatch: row != ncon");
    }

    qp.A.resize(ncon, nvar);
    qp.A.setFromTriplets(At.begin(), At.end());

    // 修正不可行行（若存在 l>u）
    for (int i = 0; i < qp.l.size(); ++i) {
      if (qp.l(i) > qp.u(i)) {
        qp.l(i) = qp.u(i) - 1e-9;
      }
    }

    return qp;
  }

  // ---------------- 性能下降评估（MATLAB 风格） ----------------
  double evaluatePerformanceDegradation(const Eigen::Matrix<double, 7, 1>& u_final,
                                        const Eigen::Matrix<double, 7, 1>& u_primary) const {
    const double change = (u_final - u_primary).norm();
    const double max_change = (params_.umax - params_.umin).norm();
    if (max_change <= 1e-9) return 1.0;
    double degr = std::min(change / max_change, 1.0);
    if (change < 0.01) degr = 0.0;
    return degr;
  }

  // ---------------- OSQP 求解封装 ----------------
  int solveOSQPPrimary(const PrimaryQP& qp, Eigen::VectorXd* sol_x) {
    if (!sol_x) return -1;

    if (!primary_initialized_) {
      primary_solver_.settings()->setVerbosity(false);
      primary_solver_.settings()->setWarmStart(true);
      primary_solver_.settings()->setMaxIteration(30000);
      primary_solver_.settings()->setAbsoluteTolerance(1e-5);
      primary_solver_.settings()->setRelativeTolerance(1e-5);
      primary_solver_.settings()->setPolish(true);

      primary_solver_.data()->setNumberOfVariables(qp.nvar);
      primary_solver_.data()->setNumberOfConstraints(qp.ncon);
      if (!primary_solver_.data()->setHessianMatrix(qp.H)) return -1;
      if (!primary_solver_.data()->setGradient(qp.g)) return -1;
      if (!primary_solver_.data()->setLinearConstraintsMatrix(qp.A)) return -1;
      if (!primary_solver_.data()->setLowerBound(qp.l)) return -1;
      if (!primary_solver_.data()->setUpperBound(qp.u)) return -1;

      if (!primary_solver_.initSolver()) return -1;
      primary_initialized_ = true;

      primary_primal_ = Eigen::VectorXd::Zero(qp.nvar);
      primary_dual_   = Eigen::VectorXd::Zero(qp.ncon);
    } else {
      // 更新 QP
      primary_solver_.updateHessianMatrix(qp.H);
      primary_solver_.updateGradient(qp.g);
      primary_solver_.updateLinearConstraintsMatrix(qp.A);
      primary_solver_.updateBounds(qp.l, qp.u);
    }

    // 热启动
    if (primary_primal_.size() == qp.nvar) primary_solver_.setWarmStart(primary_primal_);
    if (primary_dual_.size()   == qp.ncon) primary_solver_.setWarmStartDual(primary_dual_);

    const auto ret = primary_solver_.solveProblem();
    const int status = static_cast<int>(ret);

    // OsqpEigen 返回枚举：
    // 0 = NoError（已求解），其他为错误。
    // 这里把 0 视为成功（类似 MATLAB 的 status=1）
    if (status == 0) {
      *sol_x = primary_solver_.getSolution();
      primary_primal_ = *sol_x;
      primary_dual_ = primary_solver_.getDualSolution();
      return 1;
    } else {
      // 仍尝试获取当前最优解
      *sol_x = primary_solver_.getSolution();
      primary_primal_ = *sol_x;
      primary_dual_ = primary_solver_.getDualSolution();
      return -1;
    }
  }

  int solveOSQPSecondary(const SecondaryQP& qp, Eigen::VectorXd* sol_x) {
    if (!sol_x) return -1;

    if (!secondary_initialized_) {
      secondary_solver_.settings()->setVerbosity(false);
      secondary_solver_.settings()->setWarmStart(true);
      secondary_solver_.settings()->setMaxIteration(30000);
      secondary_solver_.settings()->setAbsoluteTolerance(1e-5);
      secondary_solver_.settings()->setRelativeTolerance(1e-5);
      secondary_solver_.settings()->setPolish(true);

      secondary_solver_.data()->setNumberOfVariables(qp.nvar);
      secondary_solver_.data()->setNumberOfConstraints(qp.ncon);
      if (!secondary_solver_.data()->setHessianMatrix(qp.H)) return -1;
      if (!secondary_solver_.data()->setGradient(qp.g)) return -1;
      if (!secondary_solver_.data()->setLinearConstraintsMatrix(qp.A)) return -1;
      if (!secondary_solver_.data()->setLowerBound(qp.l)) return -1;
      if (!secondary_solver_.data()->setUpperBound(qp.u)) return -1;

      if (!secondary_solver_.initSolver()) return -1;
      secondary_initialized_ = true;

      secondary_primal_ = Eigen::VectorXd::Zero(qp.nvar);
      secondary_dual_   = Eigen::VectorXd::Zero(qp.ncon);
    } else {
      secondary_solver_.updateHessianMatrix(qp.H);
      secondary_solver_.updateGradient(qp.g);
      secondary_solver_.updateLinearConstraintsMatrix(qp.A);
      secondary_solver_.updateBounds(qp.l, qp.u);
    }

    if (secondary_primal_.size() == qp.nvar) secondary_solver_.setWarmStart(secondary_primal_);
    if (secondary_dual_.size()   == qp.ncon) secondary_solver_.setWarmStartDual(secondary_dual_);

    const auto ret = secondary_solver_.solveProblem();
    const int status = static_cast<int>(ret);

    if (status == 0) {
      *sol_x = secondary_solver_.getSolution();
      secondary_primal_ = *sol_x;
      secondary_dual_ = secondary_solver_.getDualSolution();
      return 1;
    } else {
      *sol_x = secondary_solver_.getSolution();
      secondary_primal_ = *sol_x;
      secondary_dual_ = secondary_solver_.getDualSolution();
      return -1;
    }
  }

private:
  mutable std::mutex mutex_;
  GMPCParams params_;

  // OSQP 求解器与热启动缓存
  OsqpEigen::Solver primary_solver_;
  OsqpEigen::Solver secondary_solver_;
  bool primary_initialized_ = false;
  bool secondary_initialized_ = false;
  Eigen::VectorXd primary_primal_;
  Eigen::VectorXd primary_dual_;
  Eigen::VectorXd secondary_primal_;
  Eigen::VectorXd secondary_dual_;

  // Jdot 估计缓存
  bool J_prev_valid_ = false;
  Eigen::Matrix<double, 6, 7> J_prev_;

  // 上一周期控制量（用于跨周期约束）
  Eigen::Matrix<double, 7, 1> last_u_cmd_ = Eigen::Matrix<double, 7, 1>::Zero();
  bool last_u_prev_valid_ = false;
};

// ------------------------------ 文件结尾提示 ------------------------------
//
// 控制器使用示例：
//   DualLayerGMPC gmpc;
//   GMPCParams p; ... 填写 ...
//   gmpc.setParams(p);
//
// 在 update() 中：
//   GMPCInput in; 填写 q、dq、位姿、目标位姿、Vd、M、coriolis、gravity、J
//   Eigen::Matrix<double,7,1> u = gmpc.computeTau(in);
//   Eigen::Matrix<double,7,1> tau_cmd = u + coriolis;  // 推荐做法
//   tau_cmd = saturateTorqueRate(tau_cmd, tau_J_d);
//   setCommand(tau_cmd)
//
// ----------------------------------------------------------------------------

}  // namespace serl_franka_controllers
