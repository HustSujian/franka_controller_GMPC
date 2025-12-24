// gmpc_dual_layer.cpp
// Dual-layer GMPC (Primary tracking + Secondary nullspace/smoothness) for 7-DoF torque control.
// - Primary QP: tracking in error-state y=[phi; V] with constraints
// - Secondary QP: within deviation box around primary solution, optimize smoothness + nullspace preference
//
// This implementation is adapted from the uploaded MATLAB "7dof-双层.txt":
//   - solvePrimaryMPC() + buildPrimaryConstraints() + buildPrimaryCost()
//   - solveSecondaryMPC_Modified() + deviation constraints + smooth + N'RnullN
//
// And follows OsqpEigen usage patterns similar to uploaded Jacobian.cpp.
//
// NOTE (Franka practical):
// - Franka provides mass matrix M(q), coriolis vector c(q,dq), gravity g(q) via model handle.
// - Coriolis *matrix* C(q,dq) is not available; Jdot is not directly available either.
// - We therefore use an affine Vdot model:
//     Vdot = J * M^{-1} * u  +  (Jdot*dq - J*M^{-1}(coriolis+gravity))
//   and estimate Jdot via finite difference.

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

// ------------------------------ Utilities ------------------------------

static inline Eigen::Matrix3d skew3(const Eigen::Vector3d& a) {
  Eigen::Matrix3d A;
  A << 0.0, -a.z(), a.y(),
       a.z(), 0.0, -a.x(),
      -a.y(), a.x(), 0.0;
  return A;
}

// Lie algebra adjoint operator ad_V for a twist V = [w; v] in R^6
// ad_V = [ [w^, 0],
//          [v^, w^] ]
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

// Damped pseudoinverse of J (6x7): J^T (J J^T + lambda^2 I)^-1
static inline Eigen::Matrix<double, 7, 6> dampedPinvJT(
    const Eigen::Matrix<double, 6, 7>& J, double lambda) {
  Eigen::Matrix<double, 6, 6> JJt = J * J.transpose();
  Eigen::Matrix<double, 6, 6> reg = JJt + (lambda * lambda) * Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 7, 6> Jinv = J.transpose() * reg.inverse();
  return Jinv;
}

// Compute condition number approx for J using SVD
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
  // sqrt(det(J J^T)) can be computed via SVD: product of singular values
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto& s = svd.singularValues();
  double prod = 1.0;
  for (int i = 0; i < s.size(); ++i) prod *= s(i);
  return prod;
}

// SE(3) log: input T in SE(3), output 4x4 matrix in se(3)
// We use Eigen's matrix log for simplicity (works for general matrices).
// Then extract phi as in MATLAB:
//   w = [X0(3,2); X0(1,3); X0(2,1)]   (1-indexed)
//   p = X0(1:3,4)
static inline Eigen::Matrix<double, 6, 1> se3LogPhi(const Eigen::Matrix4d& T_err) {
  Eigen::Matrix4d X0 = T_err.log();  // requires MatrixFunctions
  Eigen::Matrix<double, 6, 1> phi;
  phi(0) = X0(2,1);  // (3,2) in MATLAB
  phi(1) = X0(0,2);  // (1,3)
  phi(2) = X0(1,0);  // (2,1)
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

// Convert dense to sparse (row-major build)
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

// ------------------------------ Parameters ------------------------------

struct GMPCParams {
  // Dimensions (fixed for this implementation)
  static constexpr int Nx = 12;  // state: [phi(6); V(6)]
  static constexpr int Nu = 7;   // control: joint torques (or torque control term)
  int Nt = 10;                   // horizon length
  double dt = 0.001;             // controller dt (should match Franka loop)

  // Weights (Primary)
  Eigen::Matrix<double, 12, 12> Q = Eigen::Matrix<double, 12, 12>::Identity();
  Eigen::Matrix<double, 12, 12> P = Eigen::Matrix<double, 12, 12>::Identity();
  Eigen::Matrix<double, 7, 7>  R = 1e-8 * Eigen::Matrix<double, 7, 7>::Identity();

  // Optional delta-u weight for Primary (like MATLAB param.R_delta)
  bool use_R_delta = false;
  Eigen::Matrix<double, 7, 7> R_delta = Eigen::Matrix<double, 7, 7>::Zero();
  Eigen::Matrix<double, 7, 7> R_cross = Eigen::Matrix<double, 7, 7>::Zero();

  // Control bounds
  Eigen::Matrix<double, 7, 1> umin = (-50.0) * Eigen::Matrix<double, 7, 1>::Ones();
  Eigen::Matrix<double, 7, 1> umax = ( 50.0) * Eigen::Matrix<double, 7, 1>::Ones();

  // Hard torque change bounds within horizon (|u_k - u_{k-1}| <= max)
  Eigen::Matrix<double, 7, 1> du_max = (20.0) * Eigen::Matrix<double, 7, 1>::Ones();

  // Cross-cycle first-step change bound around u_prev
  Eigen::Matrix<double, 7, 1> du_cross_max = (10.0) * Eigen::Matrix<double, 7, 1>::Ones();

  // State bounds (error bounds)
  // Following MATLAB style, but you should tune to your application.
  Eigen::Matrix<double, 12, 1> xmin = (-10.0) * Eigen::Matrix<double, 12, 1>::Ones();
  Eigen::Matrix<double, 12, 1> xmax = ( 10.0) * Eigen::Matrix<double, 12, 1>::Ones();

  // "Actual" pose error bounds around target (position/orientation/velocity)
  Eigen::Matrix<double, 3, 1> xmin_rot = (-10.0) * Eigen::Matrix<double, 3, 1>::Ones();
  Eigen::Matrix<double, 3, 1> xmax_rot = ( 10.0) * Eigen::Matrix<double, 3, 1>::Ones();

  Eigen::Matrix<double, 3, 1> xmin_pos = (-0.2) * Eigen::Matrix<double, 3, 1>::Ones();
  Eigen::Matrix<double, 3, 1> xmax_pos = ( 0.2) * Eigen::Matrix<double, 3, 1>::Ones();

  Eigen::Matrix<double, 6, 1> xmin_vel = (-2.0) * Eigen::Matrix<double, 6, 1>::Ones();
  Eigen::Matrix<double, 6, 1> xmax_vel = ( 2.0) * Eigen::Matrix<double, 6, 1>::Ones();

  // Secondary layer parameters
  double alpha_tolerance = 0.95;
  double delta_deviation = 0.001;  // deviation box around primary solution
  Eigen::Matrix<double, 7, 7> R_null = Eigen::Matrix<double, 7, 7>::Identity();
  double w_smooth = 1e-8;
  double w_null = 1e-6;
};

// ------------------------------ IO struct ------------------------------

struct GMPCInput {
  // Current robot state
  Eigen::Matrix<double, 7, 1> q;
  Eigen::Matrix<double, 7, 1> dq;

  // Current EE pose in base frame (O frame in Franka)
  Eigen::Quaterniond orientation;
  Eigen::Vector3d position;

  // Desired EE pose
  Eigen::Quaterniond orientation_d;
  Eigen::Vector3d position_d;

  // Desired twist Vd in base frame (6x1)
  Eigen::Matrix<double, 6, 1> Vd;

  // Model terms
  Eigen::Matrix<double, 7, 7> M;
  Eigen::Matrix<double, 7, 1> coriolis;
  Eigen::Matrix<double, 7, 1> gravity;

  // Jacobian (base frame) and optional Jdot estimate if you have it
  Eigen::Matrix<double, 6, 7> J;
};

// ------------------------------ Dual-layer GMPC Solver ------------------------------

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

  // Main entry: compute torque control term u (7x1).
  // Recommended usage in CartesianImpedanceController:
  //   u = gmpc.computeTau(in);
  //   tau_cmd = u + coriolis; (and then saturateTorqueRate)
  Eigen::Matrix<double, 7, 1> computeTau(const GMPCInput& in) {
    std::lock_guard<std::mutex> lock(mutex_);

    // ----------- Build p0 (initial state) = [phi; V_current] -----------
    // MATLAB: X00 = X00^{-1} * Xd; X0 = logm(X00); phi from X0, then append x0(8:end) as current twist.
    // Here: phi = log( T_cur^{-1} T_des ), and V_current = J * dq
    const Eigen::Matrix4d T_cur = poseToT(in.orientation, in.position);
    const Eigen::Matrix4d T_des = poseToT(in.orientation_d, in.position_d);
    const Eigen::Matrix4d T_err = T_cur.inverse() * T_des;

    const Eigen::Matrix<double, 6, 1> phi = se3LogPhi(T_err);
    const Eigen::Matrix<double, 6, 1> V_cur = in.J * in.dq;

    Eigen::Matrix<double, 12, 1> p0;
    p0 << phi, V_cur;

    // Desired twist trajectory over horizon: we follow MATLAB xid(k,:) = desired twist samples.
    // Here we keep it constant for the horizon (can be replaced by a provided buffer).
    std::vector<Eigen::Matrix<double, 6, 1>> Vd_seq(params_.Nt);
    for (int k = 0; k < params_.Nt; ++k) Vd_seq[k] = in.Vd;

    // ----------- Adaptive damping & nullspace projector N -----------
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

    // ----------- Primary layer solve -----------
    Eigen::VectorXd sol1_x;
    int status1 = 0;
    Eigen::Matrix<double, 7, 1> u_primary = Eigen::Matrix<double, 7, 1>::Zero();

    {
      PrimaryQP qp1 = buildPrimaryQP(in, p0, Vd_seq);
      status1 = solveOSQPPrimary(qp1, &sol1_x);
      if (status1 == 1 && sol1_x.size() == qp1.nvar) {
        // Extract u_primary = first control at time k=0
        const int state_vars = (params_.Nt + 1) * GMPCParams::Nx;
        u_primary = sol1_x.segment(state_vars, GMPCParams::Nu);
      } else {
        // Fallback: if primary fails, output zero control term
        u_primary.setZero();
        // Still update last command for continuity
        last_u_cmd_ = u_primary;
        last_u_prev_valid_ = true;
        return u_primary;
      }
    }

    // ----------- Build feasible set for secondary -----------
    FeasibleSet fs;
    fs.full_solution = sol1_x;
    fs.u_primary_first = u_primary;
    fs.alpha = params_.alpha_tolerance;
    fs.deviation_bound = params_.delta_deviation;

    // ----------- Secondary layer solve -----------
    Eigen::Matrix<double, 7, 1> u_final = u_primary;
    int status2 = 0;
    Eigen::VectorXd sol2_x;

    {
      SecondaryQP qp2 = buildSecondaryQP(fs, Nproj);
      status2 = solveOSQPSecondary(qp2, &sol2_x);

      if (status2 == 1 && sol2_x.size() == qp2.nvar) {
        // sol2 is stacked [u0; u1; ... u_{Nt-1}] (Nu*Nt)
        u_final = sol2_x.segment(0, GMPCParams::Nu);
        // performance degradation check (same idea as MATLAB)
        const double degr = evaluatePerformanceDegradation(u_final, u_primary);
        if (degr > (1.0 - fs.alpha)) {
          // too much degradation -> keep primary
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

  // Provide last command for cross-cycle constraint
  bool hasLastU() const { return last_u_prev_valid_; }
  Eigen::Matrix<double, 7, 1> lastU() const { return last_u_cmd_; }

private:
  // ---------------- Primary QP container ----------------
  struct PrimaryQP {
    int nvar = 0;
    int ncon = 0;
    Eigen::SparseMatrix<double> H;
    Eigen::VectorXd g;
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
  };

  // ---------------- Secondary QP container ----------------
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
    Eigen::VectorXd full_solution;          // full primary solution
    Eigen::Matrix<double, 7, 1> u_primary_first;
    double alpha = 0.95;
    double deviation_bound = 0.001;
  };

  // Build Jdot estimate
  Eigen::Matrix<double, 6, 7> estimateJdot(const Eigen::Matrix<double, 6, 7>& J, double dt) {
    Eigen::Matrix<double, 6, 7> Jdot = Eigen::Matrix<double, 6, 7>::Zero();
    if (J_prev_valid_) {
      Jdot = (J - J_prev_) / std::max(dt, 1e-6);
    }
    J_prev_ = J;
    J_prev_valid_ = true;
    return Jdot;
  }

  // ---------------- Primary QP build ----------------
  PrimaryQP buildPrimaryQP(const GMPCInput& in,
                           const Eigen::Matrix<double, 12, 1>& p0,
                           const std::vector<Eigen::Matrix<double, 6, 1>>& Vd_seq) {
    const int Nx = GMPCParams::Nx;
    const int Nu = GMPCParams::Nu;
    const int Nt = params_.Nt;
    const double dt = params_.dt;

    // Variables: X(0..Nt) (Nx*(Nt+1)) + U(0..Nt-1) (Nu*Nt)
    const int nvar = Nx * (Nt + 1) + Nu * Nt;

    // Constraint blocks:
    // 1) Dynamics: Nx*Nt
    // 2) Initial state: Nx
    // 3) State bounds for k=1..Nt: Nx*Nt
    // 4) Control bounds: Nu*Nt
    // 5) Cross-cycle first-step change: Nu (if last_u exists)
    // 6) Hard delta-u within horizon: Nu*(Nt-1)
    const bool has_prev = last_u_prev_valid_;
    const int ncon =
        (Nx * Nt) + Nx + (Nx * Nt) + (Nu * Nt) + (has_prev ? Nu : 0) + (Nu * (Nt - 1));

    PrimaryQP qp;
    qp.nvar = nvar;
    qp.ncon = ncon;

    // --------- Build H and g (cost) ---------
    // Following MATLAB buildPrimaryCost():
    // For k=1..Nt:
    //   C = I; C(7:12,1:6) = -ad(Vd_k)
    //   b = [0; Vd_k] used to form q linear term: -C'Q b
    //   State block: C'Q C (or C'P C at final)
    // Control block: R (and optional delta-u)
    //
    // We'll build sparse using triplets.
    std::vector<Eigen::Triplet<double>> Ht;
    Ht.reserve(static_cast<size_t>(nvar * 5));  // rough

    qp.g = Eigen::VectorXd::Zero(nvar);

    for (int k = 1; k <= Nt; ++k) {
      Eigen::Matrix<double, 12, 12> Cmap = Eigen::Matrix<double, 12, 12>::Identity();
      Cmap.block<6,6>(6,0) = -ad6(Vd_seq[std::min(k-1, Nt-1)]);

      Eigen::Matrix<double, 12, 12> W = (k < Nt) ? params_.Q : params_.P;
      Eigen::Matrix<double, 12, 12> Hblk = Cmap.transpose() * W * Cmap;

      Eigen::Matrix<double, 12, 1> bvec = Eigen::Matrix<double, 12, 1>::Zero();
      bvec.segment<6>(6) = Vd_seq[std::min(k-1, Nt-1)];
      Eigen::Matrix<double, 12, 1> gblk = -Cmap.transpose() * W * bvec;

      const int idx0 = k * Nx;  // X_k starts at k*Nx
      // H block
      for (int r = 0; r < Nx; ++r) {
        for (int c = 0; c < Nx; ++c) {
          const double v = Hblk(r,c);
          if (std::abs(v) > 0.0) Ht.emplace_back(idx0 + r, idx0 + c, v);
        }
      }
      // g block
      qp.g.segment(idx0, Nx) += gblk;
    }

    // Control cost: block diagonal R over U sequence
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

    // Optional delta-u cost (like MATLAB)
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
          // uk^T R uk
          for (int r = 0; r < Nu; ++r) {
            for (int c = 0; c < Nu; ++c) {
              const double v = params_.R_delta(r,c);
              if (std::abs(v) > 0.0) Ht.emplace_back(uk + r, uk + c, v);
            }
          }
          // u_{k-1}^T R u_{k-1}
          for (int r = 0; r < Nu; ++r) {
            for (int c = 0; c < Nu; ++c) {
              const double v = params_.R_delta(r,c);
              if (std::abs(v) > 0.0) Ht.emplace_back(ukm1 + r, ukm1 + c, v);
            }
          }
          // cross terms -R
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

    // Regularize to ensure PSD
    for (int i = 0; i < nvar; ++i) {
      Ht.emplace_back(i, i, 1e-9);
    }

    qp.H.resize(nvar, nvar);
    qp.H.setFromTriplets(Ht.begin(), Ht.end());

    // --------- Build constraints A, l, u ---------
    std::vector<Eigen::Triplet<double>> At;
    At.reserve(static_cast<size_t>(ncon * 10));
    qp.l = Eigen::VectorXd::Constant(ncon, -std::numeric_limits<double>::infinity());
    qp.u = Eigen::VectorXd::Constant(ncon,  std::numeric_limits<double>::infinity());

    int row = 0;

    // (1) Dynamics constraints: X_{k+1} = Ad X_k + Bd u_k + hd
    // MATLAB buildPrimaryConstraints used:
    //  Ac = [ -ad(Vd), -I;
    //         0,       H ]
    //  Bc = [ 0;
    //         F ]
    //  hc = [ Vd;
    //         b ]
    //
    // Here we use practical affine model:
    //  phi_dot = -ad(Vd)*phi + V - Vd
    //  V_dot   = J*M^{-1}*u + (Jdot*dq - J*M^{-1}(coriolis+gravity))
    //
    // so:
    //  d/dt [phi] = (-ad(Vd))*phi + I*V + (-I)*Vd
    //  d/dt [V]   = 0*phi + 0*V + (J*M^{-1})*u + b_aff
    //
    // Discretize:
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
      // V dynamics: no dependence on phi,V in this practical form (can be extended)
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

      // Constraint form:
      //  -Ad * X_k + I * X_{k+1} + (-Bd)*u_k = hd
      const int xk  = k * Nx;
      const int xk1 = (k + 1) * Nx;
      const int uk  = u_offset + k * Nu;

      // -Ad on X_k
      addBlockTriplets(At, row, xk, -Ad, 0.0);
      // +I on X_{k+1}
      addIdentityTriplets(At, row, xk1, Nx, 1.0);
      // -Bd on u_k
      addBlockTriplets(At, row, uk, -Bd, 0.0);

      qp.l.segment(row, Nx) = hd;
      qp.u.segment(row, Nx) = hd;

      row += Nx;
    }

    // (2) Initial state: X_0 = p0
    {
      addIdentityTriplets(At, row, 0, Nx, 1.0);
      qp.l.segment(row, Nx) = p0;
      qp.u.segment(row, Nx) = p0;
      row += Nx;
    }

    // (3) State bounds for k=1..Nt
    // MATLAB computed bounds relative to target pose:
    //  rot bounds -> phi(1:3), pos bounds -> phi(4:6), vel bounds -> V(1:6)
    for (int k = 1; k <= Nt; ++k) {
      const int xk = k * Nx;

      Eigen::Matrix<double, 12, 1> xmin = Eigen::Matrix<double, 12, 1>::Zero();
      Eigen::Matrix<double, 12, 1> xmax = Eigen::Matrix<double, 12, 1>::Zero();

      // phi bounds
      xmin.segment<3>(0) = params_.xmin_rot;
      xmax.segment<3>(0) = params_.xmax_rot;
      xmin.segment<3>(3) = params_.xmin_pos;
      xmax.segment<3>(3) = params_.xmax_pos;

      // V bounds
      xmin.segment<6>(6) = params_.xmin_vel;
      xmax.segment<6>(6) = params_.xmax_vel;

      // Add constraint: xmin <= X_k <= xmax
      addIdentityTriplets(At, row, xk, Nx, 1.0);
      qp.l.segment(row, Nx) = xmin;
      qp.u.segment(row, Nx) = xmax;
      row += Nx;
    }

    // (4) Control bounds for k=0..Nt-1: umin <= u_k <= umax
    for (int k = 0; k < Nt; ++k) {
      const int uk = u_offset + k * Nu;
      addIdentityTriplets(At, row, uk, Nu, 1.0);
      qp.l.segment(row, Nu) = params_.umin;
      qp.u.segment(row, Nu) = params_.umax;
      row += Nu;
    }

    // (5) Cross-cycle first-step change around last_u_cmd_
    if (has_prev) {
      const int u0 = u_offset + 0 * Nu;
      addIdentityTriplets(At, row, u0, Nu, 1.0);
      qp.l.segment(row, Nu) = last_u_cmd_ - params_.du_cross_max;
      qp.u.segment(row, Nu) = last_u_cmd_ + params_.du_cross_max;
      row += Nu;
    }

    // (6) Hard delta-u within horizon: |u_k - u_{k-1}| <= du_max, for k=1..Nt-1
    for (int k = 1; k < Nt; ++k) {
      const int uk = u_offset + k * Nu;
      const int ukm1 = u_offset + (k - 1) * Nu;

      // u_k - u_{k-1} <= du_max  and >= -du_max
      // We implement as:
      //   A * z in [l,u] with A = [I on uk, -I on ukm1]
      // so l=-du_max, u=du_max
      for (int i = 0; i < Nu; ++i) {
        At.emplace_back(row + i, uk + i, 1.0);
        At.emplace_back(row + i, ukm1 + i, -1.0);
      }
      qp.l.segment(row, Nu) = -params_.du_max;
      qp.u.segment(row, Nu) =  params_.du_max;
      row += Nu;
    }

    if (row != ncon) {
      // Safety check
      throw std::runtime_error("Primary QP constraint count mismatch: row != ncon");
    }

    qp.A.resize(ncon, nvar);
    qp.A.setFromTriplets(At.begin(), At.end());

    return qp;
  }

  // ---------------- Secondary QP build ----------------
  SecondaryQP buildSecondaryQP(const FeasibleSet& fs,
                               const Eigen::Matrix<double, 7, 7>& Nproj) {
    const int Nu = GMPCParams::Nu;
    const int Nt = params_.Nt;

    // Variables: U_seq (Nu*Nt)
    const int nvar = Nu * Nt;

    // Constraints:
    // (1) deviation box around primary u_seq: 2*Nu*Nt
    // (2) control bounds: 2*Nu*Nt
    const int ncon = 4 * Nu * Nt;

    SecondaryQP qp;
    qp.nvar = nvar;
    qp.ncon = ncon;

    // Extract primary full u sequence from full solution:
    // primary solution layout: [X(0..Nt); U(0..Nt-1)]
    const int state_vars = (params_.Nt + 1) * GMPCParams::Nx;
    const int control_vars = GMPCParams::Nu * params_.Nt;
    if (fs.full_solution.size() < state_vars + control_vars) {
      throw std::runtime_error("FeasibleSet full_solution has invalid size.");
    }
    Eigen::VectorXd u_primary_full = fs.full_solution.segment(state_vars, control_vars);

    // Reshape u_primary_full into Nu x Nt
    // MATLAB used reshape to (Nt x Nu)
    std::vector<Eigen::Matrix<double, 7, 1>> uref_seq(params_.Nt);
    for (int k = 0; k < params_.Nt; ++k) {
      uref_seq[k] = u_primary_full.segment(k * Nu, Nu);
    }

    // --------- Secondary cost: smoothness + nullspace preference ---------
    // M2 = sparse(total_vars,total_vars); q2 = zeros
    std::vector<Eigen::Triplet<double>> Ht;
    Ht.reserve(static_cast<size_t>(nvar * 8));
    qp.g = Eigen::VectorXd::Zero(nvar);

    // Smoothness:
    // k=0: (u0 - u_prev)^2 if available, else just u0^2
    // k>0: (uk - u_{k-1})^2
    const double ws = params_.w_smooth;
    if (ws > 0.0) {
      for (int k = 0; k < Nt; ++k) {
        const int uk = k * Nu;
        if (k == 0) {
          // Add ws * I on u0
          for (int i = 0; i < Nu; ++i) Ht.emplace_back(uk + i, uk + i, ws);
          if (last_u_prev_valid_) {
            qp.g.segment(uk, Nu) += -(ws * last_u_cmd_);
          }
        } else {
          const int ukm1 = (k - 1) * Nu;
          // uk^2
          for (int i = 0; i < Nu; ++i) Ht.emplace_back(uk + i, uk + i, ws);
          // ukm1^2
          for (int i = 0; i < Nu; ++i) Ht.emplace_back(ukm1 + i, ukm1 + i, ws);
          // cross -ws
          for (int i = 0; i < Nu; ++i) {
            Ht.emplace_back(uk + i, ukm1 + i, -ws);
            Ht.emplace_back(ukm1 + i, uk + i, -ws);
          }
        }
      }
    }

    // Nullspace preference: sum_k u_k^T (N^T R_null N) u_k
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

    // Regularize
    for (int i = 0; i < nvar; ++i) Ht.emplace_back(i, i, 1e-9);

    qp.H.resize(nvar, nvar);
    qp.H.setFromTriplets(Ht.begin(), Ht.end());

    // --------- Secondary constraints: deviation box + bounds ---------
    std::vector<Eigen::Triplet<double>> At;
    At.reserve(static_cast<size_t>(ncon * 2));
    qp.l = Eigen::VectorXd::Constant(ncon, -std::numeric_limits<double>::infinity());
    qp.u = Eigen::VectorXd::Constant(ncon,  std::numeric_limits<double>::infinity());

    int row = 0;
    const double delta = fs.deviation_bound;

    // (1) deviation: u <= uref + delta  and u >= uref - delta
    // We express as two sets:
    //   +I*u in [-inf, uref+delta]
    //   -I*u in [-inf, -(uref-delta)]  <=> u >= uref-delta
    for (int k = 0; k < Nt; ++k) {
      const int uk = k * Nu;
      const Eigen::Matrix<double, 7, 1> uref = uref_seq[k];

      // u <= uref + delta
      for (int i = 0; i < Nu; ++i) At.emplace_back(row + i, uk + i, 1.0);
      qp.u.segment(row, Nu) = uref.array() + delta;
      row += Nu;

      // -u <= -(uref - delta)
      for (int i = 0; i < Nu; ++i) At.emplace_back(row + i, uk + i, -1.0);
      qp.u.segment(row, Nu) = -(uref.array() - delta);
      row += Nu;
    }

    // (2) bounds: umin <= u <= umax (two-sided)
    for (int k = 0; k < Nt; ++k) {
      const int uk = k * Nu;

      // +I*u in [umin, umax]
      for (int i = 0; i < Nu; ++i) At.emplace_back(row + i, uk + i, 1.0);
      qp.l.segment(row, Nu) = params_.umin;
      qp.u.segment(row, Nu) = params_.umax;
      row += Nu;

      // -I*u in [-umax, -umin]  (redundant but matches MATLAB structure)
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

    // Fix infeasible rows (if any): l>u
    for (int i = 0; i < qp.l.size(); ++i) {
      if (qp.l(i) > qp.u(i)) {
        qp.l(i) = qp.u(i) - 1e-9;
      }
    }

    return qp;
  }

  // ---------------- Performance degradation (MATLAB style) ----------------
  double evaluatePerformanceDegradation(const Eigen::Matrix<double, 7, 1>& u_final,
                                        const Eigen::Matrix<double, 7, 1>& u_primary) const {
    const double change = (u_final - u_primary).norm();
    const double max_change = (params_.umax - params_.umin).norm();
    if (max_change <= 1e-9) return 1.0;
    double degr = std::min(change / max_change, 1.0);
    if (change < 0.01) degr = 0.0;
    return degr;
  }

  // ---------------- OSQP Solve wrappers ----------------
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
      // update QP
      primary_solver_.updateHessianMatrix(qp.H);
      primary_solver_.updateGradient(qp.g);
      primary_solver_.updateLinearConstraintsMatrix(qp.A);
      primary_solver_.updateBounds(qp.l, qp.u);
    }

    // warm start
    if (primary_primal_.size() == qp.nvar) primary_solver_.setWarmStart(primary_primal_);
    if (primary_dual_.size()   == qp.ncon) primary_solver_.setWarmStartDual(primary_dual_);

    const auto ret = primary_solver_.solveProblem();
    const int status = static_cast<int>(ret);

    // OsqpEigen returns enum; we map:
    // 0 = NoError (solved), others are errors.
    // We'll treat 0 as success (status_val == 1 in MATLAB)
    if (status == 0) {
      *sol_x = primary_solver_.getSolution();
      primary_primal_ = *sol_x;
      primary_dual_ = primary_solver_.getDualSolution();
      return 1;
    } else {
      // still try to fetch best-effort solution
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

  // OSQP solvers and warm-start caches
  OsqpEigen::Solver primary_solver_;
  OsqpEigen::Solver secondary_solver_;
  bool primary_initialized_ = false;
  bool secondary_initialized_ = false;
  Eigen::VectorXd primary_primal_;
  Eigen::VectorXd primary_dual_;
  Eigen::VectorXd secondary_primal_;
  Eigen::VectorXd secondary_dual_;

  // Jdot estimation
  bool J_prev_valid_ = false;
  Eigen::Matrix<double, 6, 7> J_prev_;

  // last command for cross-cycle constraint
  Eigen::Matrix<double, 7, 1> last_u_cmd_ = Eigen::Matrix<double, 7, 1>::Zero();
  bool last_u_prev_valid_ = false;
};

// ------------------------------ End of file ------------------------------
//
// Usage in controller (sketch):
//   DualLayerGMPC gmpc;
//   GMPCParams p; ... fill ...
//   gmpc.setParams(p);
//
// In update():
//   GMPCInput in; fill q,dq, pose, pose_d, Vd, M,coriolis,gravity,J
//   Eigen::Matrix<double,7,1> u = gmpc.computeTau(in);
//   Eigen::Matrix<double,7,1> tau_cmd = u + coriolis;  // recommended
//   tau_cmd = saturateTorqueRate(tau_cmd, tau_J_d);
//   setCommand(tau_cmd)
//
// ----------------------------------------------------------------------------

}  // namespace serl_franka_controllers
