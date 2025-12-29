/*
Reference: 
  https://github.com/frankaemika/franka_ros/blob/develop/franka_example_controllers/src/cartesian_impedance_example_controller.cpp
*/

#include <serl_franka_controllers/cartesian_impedance_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <serl_franka_controllers/pseudo_inversion.h>
#include <ros/console.h>
#include <Eigen/Dense>
#include <serl_franka_controllers/gmpc_dual_layer.h>




namespace serl_franka_controllers {

bool CartesianImpedanceController::init(hardware_interface::RobotHW* robot_hw,
                                               ros::NodeHandle& node_handle) {
  std::vector<double> cartesian_stiffness_vector;
  std::vector<double> cartesian_damping_vector;
  publisher_franka_jacobian_.init(node_handle, "franka_jacobian", 1);

  sub_equilibrium_pose_ = node_handle.subscribe(
      "equilibrium_pose", 20, &CartesianImpedanceController::equilibriumPoseCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("CartesianImpedanceController: Could not read parameter arm_id");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "CartesianImpedanceController: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceController: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "CartesianImpedanceController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  dynamic_reconfigure_compliance_param_node_ =
      ros::NodeHandle(node_handle.getNamespace() + "dynamic_reconfigure_compliance_param_node");

  dynamic_server_compliance_param_ = std::make_unique<
      dynamic_reconfigure::Server<serl_franka_controllers::compliance_paramConfig>>(

      dynamic_reconfigure_compliance_param_node_);
  dynamic_server_compliance_param_->setCallback(
      boost::bind(&CartesianImpedanceController::complianceParamCallback, this, _1, _2));

  position_d_.setZero();
  orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  position_d_target_.setZero();
  orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;

  cartesian_stiffness_.setZero();
  cartesian_damping_.setZero();

    // ===== GMPC params init =====
  gmpc_params_.Nt = 10;
  gmpc_params_.dt = 0.001; // will be overwritten in update() by period
  gmpc_.setParams(gmpc_params_);
  gmpc_.reset();


  return true;
}

void CartesianImpedanceController::starting(const ros::Time& /*time*/) {
  // compute initial velocity with jacobian and set x_attractor and q_d_nullspace
  // to initial configuration
  franka::RobotState initial_state = state_handle_->getRobotState();
  // get jacobian
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  // convert to eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));

  // set equilibrium point to current state
  position_d_ = initial_transform.translation();
  orientation_d_ = Eigen::Quaterniond(initial_transform.linear());
  position_d_target_ = initial_transform.translation();
  orientation_d_target_ = Eigen::Quaterniond(initial_transform.linear());

  // set nullspace equilibrium configuration to initial q
  q_d_nullspace_ = q_initial;

  gmpc_.reset();
  J_prev_valid_ = false;
  t_total_ = 0.0;  // 重置时间计数器
}

void CartesianImpedanceController::update(const ros::Time& time,
                                                 const ros::Duration& period) {
  // get state variables
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
  jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  publishZeroJacobian(time);
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian_raw(jacobian_array.data());
  
  // 雅可比矩阵顺序调整：Franka原始顺序为[线速度3行; 角速度3行]
  // GMPC需要的顺序：[角速度3行; 线速度3行]（与Jacobian.cpp一致）
  Eigen::Matrix<double, 6, 7> jacobian;
  jacobian.block<3, 7>(0, 0) = jacobian_raw.block<3, 7>(3, 0);  // 角速度放到前3行
  jacobian.block<3, 7>(3, 0) = jacobian_raw.block<3, 7>(0, 0);  // 线速度放到后3行
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(  // NOLINT (readability-identifier-naming)
      robot_state.tau_J_d.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.linear());

  // compute error to desired pose
  // Clip translational error
  error_.head(3) << position - position_d_;
  for (int i = 0; i < 3; i++) {
    error_(i) = std::min(std::max(error_(i), translational_clip_min_(i)), translational_clip_max_(i));
  }

  // orientation error
  if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  // "difference" quaternion
  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
  error_.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  // Transform to base frame
  // Clip rotation error
  error_.tail(3) << -transform.linear() * error_.tail(3);
    for (int i = 0; i < 3; i++) {
    error_(i+3) = std::min(std::max(error_(i+3), rotational_clip_min_(i)), rotational_clip_max_(i));
  }

  error_i.head(3) << (error_i.head(3) + error_.head(3)).cwiseMax(-0.1).cwiseMin(0.1);
  error_i.tail(3) << (error_i.tail(3) + error_.tail(3)).cwiseMax(-0.3).cwiseMin(0.3);

  // compute control
  // allocate variables
  Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7);

  // pseudoinverse for nullspace handling
  // kinematic pseuoinverse
  Eigen::MatrixXd jacobian_transpose_pinv;
  pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

  tau_task << jacobian.transpose() *
                  (-cartesian_stiffness_ * error_ - cartesian_damping_ * (jacobian * dq) - Ki_ * error_i);

  Eigen::Matrix<double, 7, 1> dqe;
  Eigen::Matrix<double, 7, 1> qe;

  qe << q_d_nullspace_ - q;
  qe.head(1) << qe.head(1) * joint1_nullspace_stiffness_;
  dqe << dq;
  dqe.head(1) << dqe.head(1) * 2.0 * sqrt(joint1_nullspace_stiffness_);
  tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
                    jacobian.transpose() * jacobian_transpose_pinv) *
                       (nullspace_stiffness_ * qe -
                        (2.0 * sqrt(nullspace_stiffness_)) * dqe);

  // ====== Build GMPC inputs ======
  // 获取Franka动力学参数（通过 franka_hw::FrankaModelHandle）
  // 与 Jacobian.cpp 不同：
  //   - Jacobian.cpp 使用 Pinocchio: computeMinverse(), computeCoriolisMatrix(), computeGeneralizedGravity()
  //   - 本代码使用 franka_ros: model_handle_->getMass(), getCoriolis(), getGravity()
  // 优点：franka_ros 提供的参数基于真实机器人标定，精度更高
  std::array<double, 49> mass_array = model_handle_->getMass();       // 7x7 质量矩阵 M(q)
  std::array<double, 7> gravity_array = model_handle_->getGravity();  // 7x1 重力向量 G(q)

  Eigen::Matrix<double, 7, 7> M = Eigen::Map<Eigen::Matrix<double, 7, 7>>(mass_array.data());
  Eigen::Matrix<double, 7, 1> G = Eigen::Map<Eigen::Matrix<double, 7, 1>>(gravity_array.data());

  // 获取实际控制周期
  double dt = 0.001;
  if (period.toSec() > 1e-6 && period.toSec() < 0.1)
    dt = period.toSec();

  // 计算Jdot（有限差分法）
  // 注意：Franka 不提供 Jdot 接口，也不支持 Pinocchio
  // Jacobian.cpp 使用：pinocchio::computeJointJacobiansTimeVariation(model, data, q, dq)
  // 这里使用有限差分：Jdot ≈ (J_current - J_prev) / dt
  Eigen::Matrix<double, 6, 7> Jdot = Eigen::Matrix<double, 6, 7>::Zero();
  if (J_prev_valid_)
  {
    Jdot = (jacobian - J_prev_) / dt;
  }
  J_prev_ = jacobian;
  J_prev_valid_ = true;

  // 构造期望状态（13维：[qw qx qy qz px py pz wx wy wz vx vy vz]）
  serl_franka_controllers::DesiredState13 xd0;
  xd0.v.setZero();
  
  if (use_trajectory_) {
    // 使用螺旋轨迹生成（参考 Jacobian.cpp）
    // 注意：这里的轨迹生成函数在 gmpc_dual_layer.cpp 中是 static 的
    // 因此我们在这里直接计算当前期望状态
    double T = 30.0;  // 总周期
    double t_in_period = std::fmod(t_total_, T);
    
    double radius = 0.4;
    double height = 0.2;
    double turns = 0.6;
    double t_end = 30.0;
    
    // 位置（螺旋轨迹）
    double x = radius * std::cos(2 * M_PI * turns * t_in_period / t_end);
    double y = 0 + radius * std::sin(2 * M_PI * turns * t_in_period / t_end);
    double z = 0.9 + height * t_in_period / t_end;
    
    // 速度
    double x_d = -radius * (2 * M_PI * turns) / t_end * std::sin(2 * M_PI * turns * t_in_period / t_end);
    double y_d =  radius * (2 * M_PI * turns) / t_end * std::cos(2 * M_PI * turns * t_in_period / t_end);
    double z_d = height / t_end;
    
    // 四元数（保持恒定姿态）
    Eigen::Vector4d q0(0.6, 0, 0, 0.8);
    q0.normalize();
    
    xd0.v(0) = q0(0);  // qw
    xd0.v(1) = q0(1);  // qx
    xd0.v(2) = q0(2);  // qy
    xd0.v(3) = q0(3);  // qz
    xd0.v(4) = x;      // px
    xd0.v(5) = y;      // py
    xd0.v(6) = z;      // pz
    xd0.v(7) = 0;      // wx
    xd0.v(8) = 0;      // wy
    xd0.v(9) = 0;      // wz
    xd0.v(10) = x_d;   // vx
    xd0.v(11) = y_d;   // vy
    xd0.v(12) = z_d;   // vz
    
    ROS_INFO_THROTTLE(2.0, "Trajectory mode | t=%.2f | pos=[%.3f, %.3f, %.3f] | vel=[%.3f, %.3f, %.3f]",
                      t_in_period, x, y, z, x_d, y_d, z_d);
  } else {
    // 固定点跟踪模式
    xd0.v(0) = orientation_d_.w();
    xd0.v(1) = orientation_d_.x();
    xd0.v(2) = orientation_d_.y();
    xd0.v(3) = orientation_d_.z();
    xd0.v(4) = position_d_(0);
    xd0.v(5) = position_d_(1);
    xd0.v(6) = position_d_(2);
    // xd0.v(7..12) = 0 (已经通过 setZero() 设置)
  }

  // 更新GMPC时间步长
  gmpc_.setDt(dt);

  // 调用GMPC求解器
  Eigen::Matrix<double, 7, 1> tau_u = Eigen::Matrix<double, 7, 1>::Zero();
  bool ok = gmpc_.computeTauMPC(
      transform,    // 当前末端位姿
      q, dq,        // 关节位置和速度
      jacobian,     // 雅可比矩阵
      Jdot,         // 雅可比导数
      M,            // 质量矩阵
      coriolis,     // 科氏力+重力（Franka的coriolis已包含重力）
      G,            // 重力向量（单独传递用于GMPC内部计算）
      xd0,          // 期望状态
      &tau_u);      // 输出：控制力矩（不含补偿项）

  // 组装最终控制指令
  Eigen::Matrix<double, 7, 1> tau_d_cmd;
  if (!ok)
  {
    // GMPC求解失败，降级到原阻抗控制器
    ROS_WARN_THROTTLE(1.0, "GMPC solver failed, using fallback impedance control");
    tau_d_cmd = tau_task + tau_nullspace + coriolis;
  }
  else
  {
    // GMPC成功，使用GMPC输出
    // 注意：tau_u是纯控制项，需要加上coriolis补偿（包含重力）
    tau_d_cmd = tau_u + coriolis;
    
    // 更新时间（用于轨迹生成）
    t_total_ += dt;
    
    // 调试信息（每秒打印一次）
    ROS_INFO_THROTTLE(1.0, "GMPC active | tau_norm: %.3f | position_error: [%.3f, %.3f, %.3f]",
                      tau_u.norm(),
                      error_(0), error_(1), error_(2));
  }

  // 力矩变化率限制（安全机制）
  tau_d_cmd = saturateTorqueRate(tau_d_cmd, tau_J_d);

  // 发送控制指令到各关节
  for (size_t i = 0; i < 7; ++i)
  {
    joint_handles_[i].setCommand(tau_d_cmd(i));
  }

  // update parameters changed online either through dynamic reconfigure or through the interactive
  // target by filtering
  cartesian_stiffness_ =
      filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * cartesian_stiffness_;
  cartesian_damping_ =
      filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * cartesian_damping_;
  nullspace_stiffness_ =
      filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;
  joint1_nullspace_stiffness_ =
      filter_params_ * joint1_nullspace_stiffness_target_ + (1.0 - filter_params_) * joint1_nullspace_stiffness_;
  position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
  orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);
  Ki_ = filter_params_ * Ki_target_ + (1.0 - filter_params_) * Ki_;
}

void CartesianImpedanceController::publishZeroJacobian(const ros::Time& time) {
  if (publisher_franka_jacobian_.trylock()) {
      for (size_t i = 0; i < jacobian_array.size(); i++) {
        publisher_franka_jacobian_.msg_.zero_jacobian[i] = jacobian_array[i];
      }
      publisher_franka_jacobian_.unlockAndPublish();
    }
}


Eigen::Matrix<double, 7, 1> CartesianImpedanceController::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] =
        tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}

void CartesianImpedanceController::complianceParamCallback(
    serl_franka_controllers::compliance_paramConfig& config,
    uint32_t /*level*/) {
  cartesian_stiffness_target_.setIdentity();
  cartesian_stiffness_target_.topLeftCorner(3, 3)
      << config.translational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_stiffness_target_.bottomRightCorner(3, 3)
      << config.rotational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.setIdentity();
  // Damping ratio = 1
  cartesian_damping_target_.topLeftCorner(3, 3)
      << config.translational_damping * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.bottomRightCorner(3, 3)
      << config.rotational_damping * Eigen::Matrix3d::Identity();
 
  nullspace_stiffness_target_ = config.nullspace_stiffness;
  joint1_nullspace_stiffness_target_ = config.joint1_nullspace_stiffness;

  translational_clip_min_ << -config.translational_clip_neg_x, -config.translational_clip_neg_y, -config.translational_clip_neg_z;
  translational_clip_max_ << config.translational_clip_x, config.translational_clip_y, config.translational_clip_z;
  rotational_clip_min_ << -config.rotational_clip_neg_x, -config.rotational_clip_neg_y, -config.rotational_clip_neg_z;
  rotational_clip_max_ << config.rotational_clip_x, config.rotational_clip_y, config.rotational_clip_z;

  Ki_target_.setIdentity();
  Ki_target_.topLeftCorner(3, 3)
      << config.translational_Ki * Eigen::Matrix3d::Identity();
  Ki_target_.bottomRightCorner(3, 3)
      << config.rotational_Ki * Eigen::Matrix3d::Identity();
}


// --------- small helpers ----------
static inline Eigen::Matrix3d hat3(const Eigen::Vector3d& w) {
  Eigen::Matrix3d W;
  W <<    0.0, -w.z(),  w.y(),
       w.z(),    0.0, -w.x(),
      -w.y(),  w.x(),   0.0;
  return W;
}

// so(3) log: R -> phi (axis-angle vector)
static inline Eigen::Vector3d so3Log(const Eigen::Matrix3d& R) {
  double cos_theta = (R.trace() - 1.0) * 0.5;
  cos_theta = std::min(1.0, std::max(-1.0, cos_theta));
  double theta = std::acos(cos_theta);

  if (theta < 1e-9) {
    // near zero: log(R) ~ vee(R - R^T)/2
    Eigen::Vector3d w;
    w << (R(2,1) - R(1,2)),
         (R(0,2) - R(2,0)),
         (R(1,0) - R(0,1));
    return 0.5 * w;
  } else {
    Eigen::Vector3d w;
    w << (R(2,1) - R(1,2)),
         (R(0,2) - R(2,0)),
         (R(1,0) - R(0,1));
    w *= 0.5 / std::sin(theta);
    return theta * w;
  }
}

// left Jacobian inverse for SO(3): J^{-1}(phi)
static inline Eigen::Matrix3d so3LeftJacInv(const Eigen::Vector3d& phi) {
  double theta = phi.norm();
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  if (theta < 1e-9) {
    // series
    Eigen::Matrix3d A = hat3(phi);
    return I + 0.5 * A + (1.0/12.0) * (A*A);
  }
  Eigen::Matrix3d A = hat3(phi);
  double half = 0.5 * theta;
  double cot_half = std::cos(half) / std::sin(half);
  return I + 0.5 * A + (1.0 - theta * cot_half / 2.0) / (theta*theta) * (A*A);
}

// se(3) log: g = [R p; 0 1] -> xi = [phi; rho]
static inline Eigen::Matrix<double,6,1> se3Log(const Eigen::Matrix3d& R, const Eigen::Vector3d& p) {
  Eigen::Matrix<double,6,1> xi;
  Eigen::Vector3d phi = so3Log(R);
  Eigen::Matrix3d Jinv = so3LeftJacInv(phi);
  Eigen::Vector3d rho = Jinv * p;   // this matches "log map" translation part in twist coordinates
  xi.head<3>() = phi;
  xi.tail<3>() = rho;
  return xi;
}

// build p0 = [phi(6); V(6)]  (12x1)
// g_current = O_T_EE (base/world -> EE) in franka; choose consistent with your Xd
static inline Eigen::Matrix<double,12,1> build_p0_leftInvariant(
    const Eigen::Matrix4d& g_current,
    const Eigen::Matrix4d& g_desired,
    const Eigen::Matrix<double,6,1>& V_current) {

  // left-invariant error: g_e = g^{-1} g_d
  Eigen::Matrix4d g_e = g_current.inverse() * g_desired;
  Eigen::Matrix3d R_e = g_e.block<3,3>(0,0);
  Eigen::Vector3d p_e = g_e.block<3,1>(0,3);

  Eigen::Matrix<double,6,1> phi = se3Log(R_e, p_e); // [rot; trans] in se(3) log coordinates

  Eigen::Matrix<double,12,1> p0;
  p0.head<6>() = phi;
  p0.tail<6>() = V_current; // 你 MATLAB 里 x0(8:end)
  return p0;
}

void CartesianImpedanceController::equilibriumPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& msg) {
  position_d_target_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  error_i.setZero();
  Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
  orientation_d_target_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
      msg->pose.orientation.z, msg->pose.orientation.w;
  if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
    orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
  }
}

}  // namespace serl_franka_controllers

PLUGINLIB_EXPORT_CLASS(serl_franka_controllers::CartesianImpedanceController,
                       controller_interface::ControllerBase)
