#include <serl_franka_controllers/gmpc_dual_layer.h>

#include <cmath>
#include <iostream>
#include <limits>

namespace serl_franka_controllers {

GMPCDualLayer::GMPCDualLayer() {}

void GMPCDualLayer::reset() {
  solver_initialized_ = false;
  lower_initialized_  = false;
  primal_last_.resize(0);
  dual_last_.resize(0);
  u_prev_cycle_.setZero();
}

void GMPCDualLayer::setParams(const GMPCParams& p) {
  p_ = p;
}

bool GMPCDualLayer::initSolver() {
  const int Nx = p_.Nx;
  const int Nu = p_.Nu;
  const int Nt = p_.Nt;

  // Jacobian.cpp 的决策变量维度（原封结构）:
  // totalDim = Nx*(Nt+1) + Nu*Nt + Nx*Nt
  // 这里最后的 Nx*Nt 你 Jacobian.cpp 也保留了（虽然没有显式用到）
  const int totalDim = Nx*(Nt+1) + Nu*Nt + Nx*Nt;
  const int n = totalDim;
  const int m = n;  // Jacobian.cpp: m = n

  solver_upper_.settings()->setWarmStart(true);
  solver_upper_.settings()->setVerbosity(false);
  solver_upper_.settings()->setRelativeTolerance(1e-5);
  solver_upper_.settings()->setAdaptiveRho(true);
  solver_upper_.settings()->setMaxIteration(5000);

  solver_upper_.data()->setNumberOfVariables(n);
  solver_upper_.data()->setNumberOfConstraints(m);

  // allocate sparse with diagonal zeros
  H_.resize(n,n);
  {
    std::vector<Eigen::Triplet<double>> tri;
    tri.reserve(n);
    for(int i=0;i<n;i++) tri.emplace_back(i,i,0.0);
    H_.setFromTriplets(tri.begin(), tri.end());
  }

  A_.resize(m,n);
  {
    std::vector<Eigen::Triplet<double>> tri;
    tri.reserve(std::min(m,n));
    for(int i=0;i<std::min(m,n);i++) tri.emplace_back(i,i,0.0);
    A_.setFromTriplets(tri.begin(), tri.end());
  }

  q_ = Eigen::VectorXd::Zero(n);
  l_ = Eigen::VectorXd::Zero(m);
  u_ = Eigen::VectorXd::Zero(m);

  if (!solver_upper_.data()->setHessianMatrix(H_)) return false;
  if (!solver_upper_.data()->setGradient(q_)) return false;
  if (!solver_upper_.data()->setLinearConstraintsMatrix(A_)) return false;
  if (!solver_upper_.data()->setLowerBound(l_)) return false;
  if (!solver_upper_.data()->setUpperBound(u_)) return false;

  if (!solver_upper_.initSolver()) return false;

  primal_last_ = Eigen::VectorXd::Zero(n);
  dual_last_   = Eigen::VectorXd::Zero(m);

  solver_initialized_ = true;
  return true;
}

// ======== math helpers (ported from Jacobian.cpp) ========

Eigen::Matrix3d GMPCDualLayer::skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m << 0, -v(2), v(1),
       v(2), 0, -v(0),
      -v(1), v(0), 0;
  return m;
}

Eigen::Matrix<double,6,6> GMPCDualLayer::calculateAdjoint(const Eigen::Matrix<double,6,1>& x) {
  Eigen::Matrix<double,6,6> adj;
  Eigen::Vector3d w = x.head<3>();
  Eigen::Vector3d v = x.tail<3>();

  adj.block<3,3>(0,0) = skew(w);
  adj.block<3,3>(0,3) = Eigen::Matrix3d::Zero();
  adj.block<3,3>(3,0) = skew(v);
  adj.block<3,3>(3,3) = skew(w);
  return adj;
}

Eigen::Matrix4d GMPCDualLayer::matrix_logarithm(const Eigen::Matrix4d& X) {
  Eigen::Matrix3d R = X.block<3,3>(0,0);
  Eigen::Vector3d t = X.block<3,1>(0,3);

  Eigen::Matrix4d logX = Eigen::Matrix4d::Zero();

  const double c = (R.trace() - 1.0) / 2.0;
  double cc = std::min(1.0, std::max(-1.0, c));
  double theta = std::acos(cc);

  if (theta < 1e-10) {
    logX.block<3,3>(0,0) = Eigen::Matrix3d::Zero();
  } else {
    logX.block<3,3>(0,0) = theta * (R - R.transpose()) / (2.0 * std::sin(theta));
  }

  // NOTE: keep the same simplification as your Jacobian.cpp: translation part = t
  logX.block<3,1>(0,3) = t;

  return logX;
}

Eigen::Matrix<double,12,1> GMPCDualLayer::calculateErrorVector(
    const DesiredState13& xd,
    const Eigen::Quaterniond& q0,
    const Eigen::Vector3d& p0,
    const Eigen::Matrix<double,6,1>& V0) {

  Eigen::Quaterniond qd(xd.v(0), xd.v(1), xd.v(2), xd.v(3));
  Eigen::Vector3d pd(xd.v(4), xd.v(5), xd.v(6));

  Eigen::Matrix3d Rd = qd.toRotationMatrix();
  Eigen::Matrix3d R0 = q0.toRotationMatrix();

  Eigen::Matrix4d Xd = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d X00= Eigen::Matrix4d::Identity();

  Xd.block<3,3>(0,0)  = Rd;
  Xd.block<3,1>(0,3)  = pd;
  X00.block<3,3>(0,0) = R0;
  X00.block<3,1>(0,3) = p0;

  Eigen::Matrix4d Xerr = X00.inverse() * Xd;
  Eigen::Matrix4d logX = matrix_logarithm(Xerr);

  Eigen::Matrix3d log_R = logX.block<3,3>(0,0);
  Eigen::Vector3d log_p = logX.block<3,1>(0,3);

  Eigen::Matrix<double,12,1> e;
  e << log_R(2,1), log_R(0,2), log_R(1,0),
       log_p(0), log_p(1), log_p(2),
       V0(0), V0(1), V0(2), V0(3), V0(4), V0(5);

  return e;
}

Eigen::MatrixXd GMPCDualLayer::solveDARE(
    const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
    const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
    int max_iter, double tol) {
  // Iterative solution (Kleinman-like fixed-point)
  Eigen::MatrixXd P = Q;
  for (int k = 0; k < max_iter; ++k) {
    Eigen::MatrixXd BtP = B.transpose() * P;
    Eigen::MatrixXd S = R + BtP * B;
    Eigen::MatrixXd K = S.ldlt().solve(BtP * A);
    Eigen::MatrixXd Pn = A.transpose() * P * (A - B * K) + Q;
    double err = (Pn - P).norm();
    P = Pn;
    if (err < tol) break;
  }
  return P;
}

Eigen::Matrix<double,7,6> GMPCDualLayer::dampedLeastSquaresJinv(
    const Eigen::Matrix<double,6,7>& J, double lambda) {
  // J_inv = J^T (J J^T + λ^2 I)^-1   (same as your MATLAB)
  Eigen::Matrix<double,6,6> JJt = J * J.transpose();
  Eigen::Matrix<double,6,6> A = JJt + (lambda*lambda) * Eigen::Matrix<double,6,6>::Identity();
  Eigen::Matrix<double,6,6> Ainv = A.inverse();
  Eigen::Matrix<double,7,6> Jinv = J.transpose() * Ainv;
  return Jinv;
}

// ======== Upper QP (ported structure from Jacobian.cpp) ========

void GMPCDualLayer::updateCostFunction(
    const Eigen::Matrix<double,12,1>& /*p0*/,
    const std::vector<DesiredState13>& xd_seq) {

  const int Nx = p_.Nx;
  const int Nu = p_.Nu;
  const int Nt = p_.Nt;

  const int totalDim = Nx*(Nt+1) + Nu*Nt + Nx*Nt;

  Eigen::SparseMatrix<double> H(totalDim, totalDim);
  Eigen::VectorXd q = Eigen::VectorXd::Zero(totalDim);

  std::vector<Eigen::Triplet<double>> tri;
  tri.reserve(Nx*Nx*Nt + Nu*Nt + Nx*Nx);

  // build stage cost blocks (Jacobian.cpp style)
  for (int k = 0; k < Nt; ++k) {
    Eigen::MatrixXd C = Eigen::MatrixXd::Identity(Nx, Nx);

    Eigen::Matrix<double,6,1> Vd;
    Vd << xd_seq[k].v(7), xd_seq[k].v(8), xd_seq[k].v(9), xd_seq[k].v(10), xd_seq[k].v(11), xd_seq[k].v(12);

    Eigen::Matrix<double,6,6> adj = calculateAdjoint(Vd);
    C.block(6,0,6,6) = -adj;

    const Eigen::MatrixXd& Qk = (k < Nt-1) ? p_.Q : p_.P;
    Eigen::MatrixXd Hx = C.transpose() * Qk * C;

    for (int i=0;i<Nx;i++) {
      for (int j=0;j<Nx;j++) {
        double val = Hx(i,j);
        if (std::abs(val) > 1e-12) tri.emplace_back(k*Nx+i, k*Nx+j, val);
      }
    }

    Eigen::VectorXd b = Eigen::VectorXd::Zero(Nx);
    b.tail(6) = Vd;
    q.segment(k*Nx, Nx) = -C.transpose() * Qk * b;
  }

  // input cost
  for (int k=0;k<Nt;k++) {
    for (int i=0;i<Nu;i++) {
      double val = p_.R(i,i);
      if (std::abs(val) > 1e-12) {
        tri.emplace_back(Nx*(Nt+1) + k*Nu + i,
                         Nx*(Nt+1) + k*Nu + i,
                         val);
      }
    }
  }

  H.setFromTriplets(tri.begin(), tri.end());

  H_ = H;
  q_ = q;

  if (!solver_upper_.updateHessianMatrix(H_)) {
    std::cerr << "[GMPC] updateHessianMatrix failed\n";
  }
  if (!solver_upper_.updateGradient(q_)) {
    std::cerr << "[GMPC] updateGradient failed\n";
  }
}

void GMPCDualLayer::updateConstraints(
    const Eigen::Matrix<double,12,1>& p0,
    const std::vector<DesiredState13>& xd_seq,
    const Eigen::Matrix<double,7,7>& M,
    const Eigen::Matrix<double,7,7>& Cmat,
    const Eigen::Matrix<double,7,1>& G,
    const Eigen::Matrix<double,6,7>& J_in,
    const Eigen::Matrix<double,6,7>& Jdot_in) {

  const int Nx = p_.Nx;
  const int Nu = p_.Nu;
  const int Nt = p_.Nt;

  const int Noff = Nx*(Nt+1);
  const int total_rows = Nx*(Nt+1) + Nu*Nt + Nx*Nt;
  const int total_cols = Nx*(Nt+1) + Nu*Nt + Nx*Nt;

  Eigen::VectorXd bmin = Eigen::VectorXd::Zero(total_rows);
  Eigen::VectorXd bmax = Eigen::VectorXd::Zero(total_rows);

  Eigen::Matrix<double,7,7> M_inv = M.inverse();

  // Jacobian.cpp 里做过行交换（把 linear / angular 顺序换一下）
  // Franka 的 zeroJacobian 通常是 [O] frame: 6x7: top=linear, bottom=angular
  // Jacobian.cpp 用的是 top=linear bottom=angular，但又做了 swap
  // 为了跟你 MATLAB/UR5 那套一致，我这里也按 Jacobian.cpp 的 swap 做：
  Eigen::Matrix<double,6,7> J = J_in;
  {
    Eigen::Matrix<double,3,7> tmp = J.block<3,7>(0,0);
    J.block<3,7>(0,0) = J.block<3,7>(3,0);
    J.block<3,7>(3,0) = tmp;
  }

  Eigen::Matrix<double,6,7> Jd = Jdot_in;
  {
    Eigen::Matrix<double,3,7> tmp = Jd.block<3,7>(0,0);
    Jd.block<3,7>(0,0) = Jd.block<3,7>(3,0);
    Jd.block<3,7>(3,0) = tmp;
  }

  // pseudo inverse (completeOrthogonalDecomposition like Jacobian.cpp)
  Eigen::Matrix<double,7,6> J_pinv = J.completeOrthogonalDecomposition().pseudoInverse();

  const Eigen::Matrix<double,6,7> J_M_inv = J * M_inv;
  const Eigen::Matrix<double,6,1> neg_J_M_inv_G = -J_M_inv * G;
  const Eigen::Matrix<double,7,7> I7 = Eigen::Matrix<double,7,7>::Identity();

  // Jacobian.cpp: H = J*M^{-1}*(M*J_inv*Jd*J_inv - C*J_inv)
  // 这里 Cmat 是 joint coriolis matrix (你现在 franka 只有 coriolis vector; 我保留接口给你自己替换)
  const Eigen::Matrix<double,7,7> term = (M * (J_pinv * Jd) * J_pinv) - (Cmat * J_pinv);
  const Eigen::Matrix<double,6,6> H = J_M_inv * term; // 6x6

  Eigen::Matrix<double,12,12> Ac = Eigen::Matrix<double,12,12>::Zero();
  Eigen::Matrix<double,12,7>  Bc = Eigen::Matrix<double,12,7>::Zero();
  Eigen::Matrix<double,12,1>  hc = Eigen::Matrix<double,12,1>::Zero();

  Bc.bottomRows(6) = J_M_inv;  // 6x7

  std::vector<Eigen::Triplet<double>> tri;
  tri.reserve( Nx*Nx*(Nt+1) + Nx*Nx*Nt + Nx*Nu*Nt + Nu*Nt + Nx*Nt );

  const Eigen::Matrix<double,12,12> I_Nx = Eigen::Matrix<double,12,12>::Identity();
  const Eigen::Matrix<double,6,6>  I6    = Eigen::Matrix<double,6,6>::Identity();

  Eigen::Matrix<double,12,12> A_sub = Eigen::Matrix<double,12,12>::Zero();
  Eigen::Matrix<double,12,7>  B_sub = Eigen::Matrix<double,12,7>::Zero();

  for (int k=0;k<Nt;k++) {
    Eigen::Matrix<double,6,1> Vd;
    Vd << xd_seq[k].v(7), xd_seq[k].v(8), xd_seq[k].v(9), xd_seq[k].v(10), xd_seq[k].v(11), xd_seq[k].v(12);

    Ac.topLeftCorner(6,6)  = -calculateAdjoint(Vd);
    Ac.topRightCorner(6,6) = -I6;
    Ac.bottomRightCorner(6,6) = H;

    hc.head(6) = Vd;
    hc.tail(6) = neg_J_M_inv_G;

    Eigen::Matrix<double,12,12> Ad = I_Nx + Ac * p_.dt;
    Eigen::Matrix<double,12,7>  Bd = Bc * p_.dt;
    Eigen::Matrix<double,12,1>  hd = hc * p_.dt;

    if (k==0) { A_sub = Ad; B_sub = Bd; }

    // dynamics: -Ad * x_k + I * x_{k+1} -Bd * u_k = hd
    for (int i=0;i<Nx;i++) {
      for (int j=0;j<Nx;j++) {
        double val = -Ad(i,j);
        if (std::abs(val) > 1e-10) tri.emplace_back(k*Nx+i, k*Nx+j, val);
      }
    }
    for (int i=0;i<Nx;i++) tri.emplace_back(k*Nx+i, (k+1)*Nx+i, 1.0);

    for (int i=0;i<Nx;i++) {
      for (int j=0;j<Nu;j++) {
        double val = -Bd(i,j);
        if (std::abs(val) > 1e-10) tri.emplace_back(k*Nx+i, Noff + k*Nu + j, val);
      }
    }

    // state bounds rows (same as Jacobian.cpp): I * x_{k+1}
    for (int i=0;i<Nx;i++) {
      tri.emplace_back(Noff + Nu*Nt + k*Nx + i, (k+1)*Nx + i, 1.0);
    }

    // input bounds rows: I * u_k
    for (int i=0;i<Nu;i++) {
      tri.emplace_back(Noff + k*Nu + i, Noff + k*Nu + i, 1.0);
    }

    bmin.segment(k*Nx, Nx) = hd;
    bmax.segment(k*Nx, Nx) = hd;
    bmin.segment(Noff + Nu*Nt + k*Nx, Nx) = p_.xmin;
    bmax.segment(Noff + Nu*Nt + k*Nx, Nx) = p_.xmax;
  }

  // initial state constraint: x0 = p0   (row Nt*Nx ... Nt*Nx+Nx-1)
  for (int i=0;i<Nx;i++) tri.emplace_back(Nt*Nx + i, i, 1.0);
  bmin.segment(Nt*Nx, Nx) = p0;
  bmax.segment(Nt*Nx, Nx) = p0;

  // input bounds stacked
  for (int k=0;k<Nt;k++) {
    bmin.segment(Noff + k*Nu, Nu) = p_.umin;
    bmax.segment(Noff + k*Nu, Nu) = p_.umax;
  }

  Eigen::SparseMatrix<double> A(total_rows, total_cols);
  A.setFromTriplets(tri.begin(), tri.end());
  A_ = A;
  l_ = bmin;
  u_ = bmax;

  if (!solver_upper_.updateLinearConstraintsMatrix(A_)) {
    std::cerr << "[GMPC] updateLinearConstraintsMatrix failed\n";
  }
  if (!solver_upper_.updateBounds(l_, u_)) {
    std::cerr << "[GMPC] updateBounds failed\n";
  }

  // Update terminal P (idare-like): from your MATLAB param.P = idare(-A,-B,Q,R)
  // Here we compute P from discrete (Ad,Bd) with DARE:
  Eigen::MatrixXd Ad_d = A_sub;
  Eigen::MatrixXd Bd_d = B_sub;

  p_.P = solveDARE(Ad_d, Bd_d, p_.Q, p_.R, 200, 1e-8);
}

bool GMPCDualLayer::solveUpperQP(Eigen::VectorXd* full_solution, Eigen::Matrix<double,7,1>* u_primary) {
  if (!solver_initialized_) return false;

  // warm start
  if (primal_last_.size() == solver_upper_.data()->getNumberOfVariables()) {
    solver_upper_.setWarmStart(true);
    solver_upper_.warmStart(primal_last_, dual_last_);
  }

  auto flag = solver_upper_.solveProblem();
  if (flag != OsqpEigen::ErrorExitFlag::NoError) {
    std::cerr << "[GMPC] OSQP upper solve failed\n";
    return false;
  }

  Eigen::VectorXd sol = solver_upper_.getSolution();
  Eigen::VectorXd dual= solver_upper_.getDualSolution();

  primal_last_ = sol;
  dual_last_   = dual;

  *full_solution = sol;

  const int Nx = p_.Nx;
  const int Nu = p_.Nu;
  const int Nt = p_.Nt;

  const int startU = Nx*(Nt+1);
  *u_primary = sol.segment(startU, Nu);
  return true;
}

// ======== Lower QP (ported from your MATLAB solveSecondaryMPC_Modified) ========

bool GMPCDualLayer::solveSecondaryQP(
    const Eigen::VectorXd& upper_full_solution,
    const Eigen::Matrix<double,7,1>& u_primary_first,
    const Eigen::Matrix<double,7,1>& u_prev,
    const Eigen::Matrix<double,6,7>& J,
    Eigen::Matrix<double,7,1>* u_final_first) {

  const int Nu = p_.Nu;
  const int Nt = p_.Nt;

  // Extract full primary control sequence from upper solution
  const int Nx = p_.Nx;
  const int startU = Nx*(Nt+1);
  Eigen::VectorXd u_primary_full = upper_full_solution.segment(startU, Nu*Nt);

  // reshape to Nt x Nu (row-major)
  std::vector<Eigen::Matrix<double,7,1>> uref_seq(Nt);
  for (int k=0;k<Nt;k++) {
    uref_seq[k] = u_primary_full.segment(k*Nu, Nu);
  }

  // N projector (your MATLAB: J_inv = J' * inv(JJ'+λ^2I), N=I-J_inv*J)
  // adaptive lambda (same idea as MATLAB: cond/manipulability)
  double manipulability = std::sqrt(std::max(1e-12, (J*J.transpose()).determinant()));
  double condJ = 1e6;
  {
    Eigen::JacobiSVD<Eigen::Matrix<double,6,7>> svd(J);
    double smin = svd.singularValues().tail(1)(0);
    double smax = svd.singularValues()(0);
    if (smin > 1e-12) condJ = smax/smin;
  }

  double lambda = p_.lambda_dls;
  if (condJ > 10.0) lambda = p_.lambda_dls * condJ / 10.0;
  else if (manipulability < 0.01) lambda = p_.lambda_dls * (0.01 / manipulability);
  lambda = std::min(lambda, 0.5);

  Eigen::Matrix<double,7,6> Jinv = dampedLeastSquaresJinv(J, lambda);
  Eigen::Matrix<double,7,7> Nproj = Eigen::Matrix<double,7,7>::Identity() - Jinv * J;
  Eigen::Matrix<double,7,7> Nw = Nproj.transpose() * p_.R_null * Nproj;

  // decision variable: U = [u0;u1;...;u_{Nt-1}]  size=Nu*Nt
  const int n = Nu*Nt;

  // Build Hessian (dense then sparse) for:
  // w_smooth*Σ||u_k-u_{k-1}||^2_{R_smooth} + w_null*Σ u_k^T Nw u_k
  Eigen::MatrixXd Hd = Eigen::MatrixXd::Zero(n,n);
  Eigen::VectorXd q  = Eigen::VectorXd::Zero(n);

  // smoothness
  for (int k=0;k<Nt;k++) {
    int si = k*Nu;
    Hd.block(si,si,Nu,Nu) += p_.w_smooth * p_.R_smooth;

    if (k==0) {
      if (u_prev.size()==Nu) {
        q.segment(si,Nu) += (-p_.w_smooth) * (p_.R_smooth * u_prev);
      }
    } else {
      int pi = (k-1)*Nu;
      Hd.block(pi,pi,Nu,Nu) += p_.w_smooth * p_.R_smooth;
      Hd.block(si,pi,Nu,Nu) += (-p_.w_smooth) * p_.R_smooth;
      Hd.block(pi,si,Nu,Nu) += (-p_.w_smooth) * p_.R_smooth;
    }
  }

  // null preference
  for (int k=0;k<Nt;k++) {
    int si = k*Nu;
    Hd.block(si,si,Nu,Nu) += p_.w_null * Nw;
  }

  Hd += 1e-6 * Eigen::MatrixXd::Identity(n,n);

  Eigen::SparseMatrix<double> P = Hd.sparseView();

  // Constraints:
  // 1) deviation: u_k in [uref_k - delta, uref_k + delta]
  // 2) bounds: u_k in [umin, umax]
  // Implement as A = [I; I], l/u stacked
  const int m = 2*n;
  Eigen::SparseMatrix<double> A(m,n);
  std::vector<Eigen::Triplet<double>> tri;
  tri.reserve(2*n);
  for (int i=0;i<n;i++) {
    tri.emplace_back(i,i,1.0);
    tri.emplace_back(i+n,i,1.0);
  }
  A.setFromTriplets(tri.begin(), tri.end());

  Eigen::VectorXd l = Eigen::VectorXd::Zero(m);
  Eigen::VectorXd u = Eigen::VectorXd::Zero(m);

  const double delta = p_.delta_deviation;

  for (int k=0;k<Nt;k++) {
    int si = k*Nu;
    l.segment(si,Nu) = uref_seq[k].array() - delta;
    u.segment(si,Nu) = uref_seq[k].array() + delta;

    l.segment(n+si,Nu) = p_.umin;
    u.segment(n+si,Nu) = p_.umax;
  }

  // Initialize lower solver if needed
  if (!lower_initialized_) {
    solver_lower_.settings()->setWarmStart(true);
    solver_lower_.settings()->setVerbosity(false);
    solver_lower_.settings()->setMaxIteration(30000);
    solver_lower_.settings()->setRelativeTolerance(1e-5);
    solver_lower_.settings()->setAdaptiveRho(true);

    solver_lower_.data()->setNumberOfVariables(n);
    solver_lower_.data()->setNumberOfConstraints(m);

    if (!solver_lower_.data()->setHessianMatrix(P)) return false;
    if (!solver_lower_.data()->setGradient(q)) return false;
    if (!solver_lower_.data()->setLinearConstraintsMatrix(A)) return false;
    if (!solver_lower_.data()->setLowerBound(l)) return false;
    if (!solver_lower_.data()->setUpperBound(u)) return false;
    if (!solver_lower_.initSolver()) return false;

    lower_initialized_ = true;
  } else {
    if (!solver_lower_.updateHessianMatrix(P)) return false;
    if (!solver_lower_.updateGradient(q)) return false;
    if (!solver_lower_.updateLinearConstraintsMatrix(A)) return false;
    if (!solver_lower_.updateBounds(l,u)) return false;
  }

  auto flag = solver_lower_.solveProblem();
  if (flag != OsqpEigen::ErrorExitFlag::NoError) {
    *u_final_first = u_primary_first;
    return false;
  }

  Eigen::VectorXd sol = solver_lower_.getSolution();
  *u_final_first = sol.segment(0,Nu);
  return true;
}

// ======== Public computeTauMPC (pure-function style) ========

bool GMPCDualLayer::computeTauMPC(
    const Eigen::Affine3d& O_T_EE,
    const Eigen::Matrix<double,7,1>& q,
    const Eigen::Matrix<double,7,1>& dq,
    const Eigen::Matrix<double,6,7>& J_in,
    const Eigen::Matrix<double,6,7>& Jdot_in,
    const Eigen::Matrix<double,7,7>& M,
    const Eigen::Matrix<double,7,1>& C,
    const Eigen::Matrix<double,7,1>& G,
    const DesiredState13& xd0,
    Eigen::Matrix<double,7,1>* tau_cmd) {

  if (!solver_initialized_) {
    if (!initSolver()) return false;
  }

  // build xd sequence (constant desired across horizon)
  std::vector<DesiredState13> xd_seq(p_.Nt);
  for (int k=0;k<p_.Nt;k++) xd_seq[k] = xd0;

  // current ee state
  Eigen::Vector3d p0 = O_T_EE.translation();
  Eigen::Quaterniond q0(O_T_EE.linear());
  q0.normalize();

  // compute current twist V0 = J*dq
  Eigen::Matrix<double,6,1> V0 = J_in * dq;

  // error state p0 (12x1)
  Eigen::Matrix<double,12,1> p0_err = calculateErrorVector(xd0, q0, p0, V0);

  // joint coriolis matrix is not available directly from franka model handle;
  // To keep full Jacobian.cpp structure we pass an approximation:
  Eigen::Matrix<double,7,7> Cmat = Eigen::Matrix<double,7,7>::Zero();
  // NOTE: You can replace this with a proper C(q,dq) if you have it elsewhere.
  // Here we only keep structure; the main feedforward is still handled by C vector outside.
  (void)C; // silence unused if you keep Cmat = 0.

  // upper QP update
  updateConstraints(p0_err, xd_seq, M, Cmat, G, J_in, Jdot_in);
  updateCostFunction(p0_err, xd_seq);

  Eigen::VectorXd full_sol;
  Eigen::Matrix<double,7,1> u_primary;
  if (!solveUpperQP(&full_sol, &u_primary)) return false;

  // lower QP
  Eigen::Matrix<double,7,1> u_final;
  bool ok_lower = solveSecondaryQP(full_sol, u_primary, u_prev_cycle_, J_in, &u_final);
  if (!ok_lower) u_final = u_primary;

  // final torque output: u_final + coriolis + gravity (保持你原来 impedance 的结构风格)
  *tau_cmd = u_final + C + G;

  u_prev_cycle_ = u_final;
  return true;
}

} // namespace serl_franka_controllers
