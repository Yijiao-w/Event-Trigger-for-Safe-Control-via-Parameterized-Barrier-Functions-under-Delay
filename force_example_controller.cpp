// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/force_example_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <franka/robot_state.h>

#include <sstream>
#include "std_msgs/String.h"

#include <fstream>

#include <string>
#include <vector>

#include <franka/robot_state.h>

#include "std_msgs/Float64MultiArray.h"  // for torque topic

// Eigen 相关
#include <Eigen/Cholesky>  // 用于 LDLT 分解
#include <Eigen/Core>

// qpOASES 求解器（需要安装 qpOASES 库）
#include <qpOASES.hpp>
USING_NAMESPACE_QPOASES;

#include <geometry_msgs/PoseStamped.h>  // 确保包含该头文件
#include <nav_msgs/Path.h>

// 时间戳
#include <std_msgs/Float64MultiArray.h>

// gp节点delay
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64.h>
#include <algorithm>

// h0数据保存
#include <iomanip>

namespace franka_example_controllers {

bool ForceExampleController::loadTrajectory(const std::string& file_path) {
  std::ifstream file(file_path.c_str());
  if (!file.is_open()) {
    ROS_ERROR_STREAM("ForceExampleController: Unable to open trajectory file: " << file_path);
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    std::istringstream stream(line);
    double time;
    char comma;

    // 2) 读 21 列到一个临时向量
    Eigen::Matrix<double, 21, 1> row;
    for (int i = 0; i < 21; ++i) {
      if (!(stream >> row(i))) {
        ROS_ERROR_STREAM("ForceController: Failed to read column " << i << " in line: " << line);
        break;
      }
    }
    // 3) 拆分成 q_des, dq_des, ddq_des
    Eigen::Matrix<double, 7, 1> q_des = row.template segment<7>(0);
    Eigen::Matrix<double, 7, 1> dq_des = row.template segment<7>(7);
    Eigen::Matrix<double, 7, 1> ddq_des = row.template segment<7>(14);
    // 4) 存入你的轨迹容器——
    //    你可以新建三个 vector: trajectory_q_, trajectory_dq_, trajectory_ddq_
    trajectory_q_.push_back(q_des);
    trajectory_dq_.push_back(dq_des);
    trajectory_ddq_.push_back(ddq_des);
  }
  file.close();

  if (trajectory_q_.empty()) {
    ROS_ERROR("ForceExampleController: Trajectory file is empty or invalid.");
    return false;
  }
  ROS_INFO_STREAM("ForceExampleController: Loaded " << trajectory_.size() << " trajectory points.");
  return true;
}

bool ForceExampleController::init(hardware_interface::RobotHW* robot_hw,
                                  ros::NodeHandle& node_handle) {
  ROS_INFO("ForceExampleController: init() function has been called!");  // 添加日志

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("ForceController: Could not read parameter arm_id");
    return false;
  }
  ROS_WARN(
      "ForceExampleController: Make sure your robot's endeffector is in contact "
      "with a horizontal surface before starting the controller!");

  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "ForceController: Invalid or no joint_names parameters provided, aborting "
        "controller init!");
    return false;
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM("ForceExampleController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (const hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM("ForceExampleController: Exception getting model handle: " << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM("ForceExampleController: Error getting state interface from hardware");
    return false;
  }

  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "ForceExampleController: Exception getting state handle from interface: " << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM("ForceExampleController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM("ForceExampleController: Exception getting joint handle: " << ex.what());
      return false;
    }
  }

  // initial subscriber
  std::string torque_topic;
  if (!node_handle.getParam("subscribe_topic", topic_name_)) {
    ROS_ERROR("ForceController: Could not read parameter subscribe_topic");
    return false;
  }

  gp_comp_sub_ = node_handle.subscribe("/gp_compensation", 10,
                                       &ForceExampleController::gpCompensationCallback, this);

  //===================delay：controller发送给gpnode xy======================
  gp_data_pub_ = node_handle.advertise<std_msgs::Float64MultiArray>("/gp_training_data", 1);

  // ---------------轨迹文件读取-----------------------------------------------
  std::string trajectory_file;
  if (!node_handle.getParam("trajectory_file", trajectory_file)) {
    ROS_ERROR("ForceExampleController: Could not read parameter trajectory_file");
    return false;
  }
  if (!loadTrajectory(trajectory_file)) {
    return false;
  }

  trajectory_index_ = 0;
  // trigger_met = false;

  //----------------------pd 控制参数-------------------------------------------------------
  debug_pub_ = node_handle.advertise<std_msgs::Float64MultiArray>("/debug_tau", 10);

  safeset_center << 0.45, 0.1, 0.56;  // 0.45,0,0.56
  safeset_radius = 0.20;              // 0.23
  hocbf_alpha1_ = 15.0;               // 15
  hocbf_alpha2_ = 20.0;               // 20

  // 初始化 J_pos_prev_ 标记（用于差分估计 J̇）
  has_prev_jacobian_ = false;

  // ---------------------------delay-----------------------------------------------------------------------
  // delay for cbf
  base_delay = 0.01;
  delay_seconds = 0.015;  // 预测控制延迟为10ms
  delay_steps = delay_seconds * 1000;

  Ld_ = 3.0;  // 原本3
  F_ = 1.5;   // 原本1.5

  // —— h0 日志设置 ——
  node_handle.param("h0_log_dir", h0_log_dir_, h0_log_dir_);
  node_handle.param("h0_log_prefix", h0_log_prefix_, h0_log_prefix_);
  node_handle.param("h0_log_decim", h0_log_decim_, h0_log_decim_);
  node_handle.param("h0_log_reserve", h0_log_reserve_, h0_log_reserve_);
  h0_pub_ = node_handle.advertise<std_msgs::Float64>("/cbf/h0", 10);
  ROS_INFO_STREAM("[H0 pub] " << h0_pub_.getTopic());  // 打印最终话题名
  plot_decim_ = 1;

  h0_log_.clear();
  h0_log_.reserve(h0_log_reserve_);
  t_log_.clear();
  t_log_.reserve(h0_log_reserve_);
  h0_log_started_ = false;
  h0_log_saved_ = false;
  // ------------------------结束delay初始化------------------------------------------------------------------

  return true;
}

void ForceExampleController::messageCallback(const std_msgs::String::ConstPtr& msg) {
  ROS_INFO_STREAM("ForceExampleController: Received message: " << msg->data);
}

void ForceExampleController::torqueCallback(const std_msgs::Float64MultiArray::ConstPtr& msg) {
  // ensure length 7
  if (msg->data.size() != 7) {
    ROS_ERROR_STREAM(
        "ForceExampleController: Received torque array of invalid size: " << msg->data.size());
    return;
  }
  // copy data to user_torque_command_ 成员变量
  for (size_t i = 0; i < 7; i++) {
    user_torque_command_[i] = msg->data[i];
  }
}

// subscribe GP miu and eta
void ForceExampleController::gpCompensationCallback(
    const std_msgs::Float64MultiArray::ConstPtr& msg) {
  if (msg->data.size() != 8) {
    ROS_ERROR("ForceExampleController: Received gp_compensation of invalid size: %lu",
              msg->data.size());
    return;
  }
  // 1-7:gp miu. 8: gp error bound
  for (size_t i = 0; i < 8; ++i) {
    pending_gp_data_[i] = msg->data[i];
  }
  has_pending_gp_ = true;
  gp_ready = true;

  trigger_met = false;
}

void ForceExampleController::starting(const ros::Time& /*time*/) {
  franka::RobotState robot_state = state_handle_->getRobotState();

  // 初始化轨迹控制时的索引
  trajectory_index_ = 0;
  //=================================delay: k=0 时发送xy给gpnode=======================
  // 等待 GP 节点订阅 /gp_training_data
  int retries = 0;
  while (gp_data_pub_.getNumSubscribers() == 0 && ros::ok()) {
    ROS_WARN_STREAM("Waiting for GP node subscriber... (" << retries++ << ")");
    ros::Duration(0.1).sleep();
  }
  ROS_INFO("GP node connected, now publishing initial training point.");

  Eigen::Matrix<double, 7, 1> current_tau_disturbance_ = Eigen::Matrix<double, 7, 1>::Zero();

  // 1. 采集 q 和 dq
  Eigen::VectorXd x(14);
  for (int i = 0; i < 7; ++i) {
    x(i) = robot_state.q[i];
    x(i + 7) = robot_state.dq[i];
  }

  // 2. 采集扰动
  //    比如 tau_disturbance = tau_cmd - tau_pd
  Eigen::Matrix<double, 7, 1> y_vec = current_tau_disturbance_;
  Eigen::VectorXd y(7);
  for (int i = 0; i < 7; ++i) {
    y(i) = y_vec(i);  // 如果没有有效扰动观测，可以先设为 0
  }

  // 3. 构造 ROS 消息
  std_msgs::Float64MultiArray msg;
  msg.data.resize(21);
  for (int i = 0; i < 14; ++i)
    msg.data[i] = x[i];
  for (int i = 0; i < 7; ++i)
    msg.data[14 + i] = y[i];

  // 4. 发布给 GP 节点
  gp_data_pub_.publish(msg);

  ROS_INFO("ForceExampleController: Published initial GP training point (k=0)");

  // trigger_met = false;
  //===================================delay结束=================================
}

void ForceExampleController::update(const ros::Time& now /*time*/, const ros::Duration& period) {
  franka::RobotState robot_state = state_handle_->getRobotState();

  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<const Eigen::Matrix<double, 7, 1>> tau_J_d(robot_state.tau_J_d.data());

  // Eigen::Map<const Eigen::Matrix<double, 7, 1>> tau_measured(robot_state.tau_J.data());

  // 2. 映射必要量
  Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());

  std::array<double, 49> M_array = model_handle_->getMass();
  std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
  std::array<double, 7> gravity_array = model_handle_->getGravity();

  Eigen::Map<const Eigen::Matrix<double, 7, 7>> M(M_array.data());
  Eigen::Map<const Eigen::Matrix<double, 7, 1>> C(coriolis_array.data());
  Eigen::Map<const Eigen::Matrix<double, 7, 1>> G(gravity_array.data());

  Eigen::Matrix<double, 7, 1> tau_pd;
  Eigen::Matrix<double, 7, 1> tau_cmd;

  double t = ros::Time::now().toSec();
  Eigen::Matrix<double, 7, 1> tau_disturbance = 1 * (0.2 * C + 0.2 * G);
  //+ 5 * sin(3 * t) * Eigen::VectorXd::Ones(7);
  Eigen::Matrix<double, 7, 1> force = C;

  // 2. 提取位置雅可比：取 jacobian 的前 3 行 (3×7)
  Eigen::Matrix<double, 3, 7> J_pos = jacobian.topRows(3);

  // 3. 用向后差分法估计 J̇（位置部分）
  Eigen::Matrix<double, 3, 7> Jdot;
  if (has_prev_jacobian_) {
    Jdot = (J_pos - J_pos_prev_) / period.toSec();  // 0.001
  } else {
    Jdot.setZero();
    has_prev_jacobian_ = true;
  }
  // 更新历史雅可比（用于下一次差分计算）
  J_pos_prev_ = J_pos;

  //********** PD 轨迹跟踪控制 **********
  if (trajectory_index_ < trajectory_q_.size()) {
    // 当前目标轨迹点
    // Eigen::Matrix<double, 7, 1> q_des = trajectory_[trajectory_index_];
    Eigen::Matrix<double, 7, 1> q_des = trajectory_q_[trajectory_index_];
    Eigen::Matrix<double, 7, 1> dq_des = trajectory_dq_[trajectory_index_];
    Eigen::Matrix<double, 7, 1> ddq_des = trajectory_ddq_[trajectory_index_];

    // 4. 计算误差和 PD 控制（目标速度设为 0）
    Eigen::Matrix<double, 7, 1> error = q_des - q;  // q 为当前实际关节位置（来自 robot_state）
    Eigen::Matrix<double, 7, 1> error_dot = dq_des - dq;  // dq 为当前实际关节速度

    Eigen::Matrix<double, 7, 1> Kp;
    Kp << 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0;

    Eigen::Matrix<double, 7, 1> Kd;
    Kd << 20.0, 20.0, 20.0, 20.0, 22.0, 20.0, 20.0;

    tau_pd = M * ddq_des.matrix() +
             (Kp.array() * error.array() + Kd.array() * error_dot.array()).matrix() + force +
             tau_disturbance - gp_compensation_;

    // 判断是否已“到达”当前目标（各关节误差小于阈值 0.1）
    if (trajectory_index_ < trajectory_q_.size() - 1) {
      trajectory_index_++;
    }

    // h0保留数据
    if (!h0_log_saved_ && trajectory_index_ >= static_cast<int>(trajectory_q_.size()) - 1) {
      saveH0LogCsv();
      h0_log_saved_ = true;
    }
  }

  // -------------------【HOCBF 功能集成】-------------------
  // 1. 获取末端位姿：利用机器人状态中齐次变换 O_T_EE（
  Eigen::Map<const Eigen::Matrix<double, 4, 4>> transform(robot_state.O_T_EE.data());
  // 提取末端位置 x (3×1)
  Eigen::Vector3d x_ee = transform.block<3, 1>(0, 3);

  // 2. 提取位置雅可比：取 jacobian 的前 3 行 (3×7)
  Eigen::Vector3d pdot1 = J_pos * dq;  // 计算dot end effector position: pdot1
  // 计算 M⁻¹：使用 Eigen 的 LDLT 分解代替 .inverse()（更稳定）
  Eigen::Matrix<double, 7, 7> M_inv = M.ldlt().solve(Eigen::Matrix<double, 7, 7>::Identity());
  Eigen::Matrix<double, 3, 1> distance_ee_center = x_ee - safeset_center;

  // 参数定义
  double epsilon = 1;  // 2 0.7  1.3
  double L_delta_alpha = 0.5;  // 0.5
  double rho = 0.07;           // 0.07
  double eta_max = 0.04;       // 0.02   0.08
  eta_delay_ = Ld_ * std::sqrt(2.0 * F_ * (delay_seconds + base_delay));
  double e_mu = eta_max + eta_delay_;
  double gamma0 = (epsilon * epsilon * e_mu * e_mu) / (hocbf_alpha1_ * hocbf_alpha2_);


  // 计算安全函数 h
  double h0 = safeset_radius * safeset_radius - distance_ee_center.squaredNorm();
  double h0_dot = -2.0 * distance_ee_center.transpose() * pdot1;

  double h1 = h0_dot + hocbf_alpha1_ * h0;
  double h1_dot = -2.0 * (distance_ee_center).transpose() * pdot1 + hocbf_alpha1_ * h0;

  double h2 = 0;


  loop_counter_++;

  if (gp_ready) {
    for (size_t i = 0; i < 7; ++i) {
      gp_compensation_(i) = pending_gp_data_[i];
    }
    eta_ = pending_gp_data_[7];
     gp_ready = false;  // reset
    gp_with_data = true;
  
  //-----------------gp trigger判断-----------------------------
  // 触发阈值
  double eta_bar = eta_ + eta_delay_;
 
  double psi_star = h0_dot + hocbf_alpha1_ * (h0 - gamma0);
  // double psi_star = h1 - (epsilon * epsilon * e_mu * e_mu) / hocbf_alpha1_ ;
  double alpha_val = hocbf_alpha2_ * psi_star;
  double delta = delay_seconds;  // 10dt

  double xi_et = -rho * alpha_val - epsilon * e_mu * e_mu +
                 (1 - rho) * epsilon * eta_bar * eta_bar + 2 * rho * L_delta_alpha * delta;

  bool trigger_met = xi_et >= 0;
  
  //----------------------Event trigger--------------------------------------------------------

  //  if (trigger_met && gp_with_data) {  // 0.01s 724 0.08s 922
   if (trigger_met) {
   // gp_with_data = false;
    trigger_met = false;
    
    trigger_count++;

    // trigger：画每个时间trigger的图
    trigger_steps_.push_back(loop_counter_);


     Eigen::VectorXd x(14), y(7);
    for (int i = 0; i < 7; ++i) {
      x(i) = robot_state.q[i];
      x(i + 7) = robot_state.dq[i];
      y(i) = tau_disturbance(i);  // 实时扰动
    }

    std_msgs::Float64MultiArray msg;
    msg.data.resize(21);
    for (int i = 0; i < 14; ++i)
      msg.data[i] = x(i);  // 接收到gp更新直接发送x

    for (int i = 0; i < 7; ++i)
      msg.data[14 + i] = y(i);
    ROS_INFO("Controller: Trigger met, sent xy to GP Node.");
    gp_data_pub_.publish(msg);
   

    ROS_INFO_STREAM("ET TRIGGERED at step " << loop_counter_ << " eta=" << eta_
                                            << " xi_et=" << xi_et << " psi_star=" << psi_star
                                            << " eta_max=" << e_mu << "delta=" << delta
                                            << "triggger" << trigger_count << "times");
   

  } else {

       Eigen::VectorXd x(14), y(7);
    for (int i = 0; i < 7; ++i) {
      x(i) = robot_state.q[i];
      x(i + 7) = robot_state.dq[i];
      y(i) = tau_disturbance(i);  // 实时扰动
    }
    std_msgs::Float64MultiArray msg;
    msg.data.resize(21);
    for (int i = 0; i < 14; ++i)
    msg.data[i] = x(i);  // 接收到gp更新直接发送x
    for (int i = 0; i < 7; ++i)
      msg.data[14 + i] = 9999.0;
    // ROS_INFO("Controller: No trigger, sent x + y=9999 to GP Node.");
    gp_data_pub_.publish(msg);
    if (loop_counter_ % 100 == 0) {
      ROS_INFO_STREAM("[Debug] loop=" << loop_counter_ << " | xi_et = " << xi_et
                                      << " | psi_star = " << psi_star
                                      << " eta_delay_=" << eta_delay_ << " alpha_val=" << alpha_val
                                      << " eta_bar=" << eta_bar << " | eta = " << eta_);
    }
  }}
    double h0_star = h0 - gamma0;

    double h0_star_dot = h0_dot;

    double h1_star = h0_star_dot + hocbf_alpha1_ * h0_star;

    // qp
    double Lf2_h = -2.0 * (pdot1.dot(pdot1)) - 2.0 * distance_ee_center.transpose() * Jdot * dq +
                   2.0 * distance_ee_center.transpose() * J_pos * M_inv * force;

    Eigen::Matrix<double, 1, 7> Lglf_h = -2.0 * distance_ee_center.transpose() * J_pos * M_inv;

    // 构造 QP 目标：使得 τ 尽量接近 tau_cmd（PD 标称控制输入）
    Eigen::RowVectorXd A_constraint = Lglf_h;
    double b_constraint = -(Lf2_h + hocbf_alpha1_ * h0_star_dot + hocbf_alpha2_ * h1_star +
                            Lglf_h * gp_compensation_ 
                            - Lglf_h.squaredNorm() / (4 * epsilon));

    Eigen::Matrix<double, 7, 7> H_qp = Eigen::Matrix<double, 7, 7>::Identity();  // 之前是2倍
    Eigen::Matrix<double, 7, 1> g_qp = -tau_pd;                                  // 之前是2倍

    // QP 不等式约束为： A * τ ≥ b_constraint

    Eigen::MatrixXd A_qp(1, 7);
    A_qp = A_constraint;  // 1x7 矩阵
   
    Eigen::Matrix<double, 1, 1> lbA(b_constraint);
    Eigen::Matrix<double, 1, 1> ubA(1e10);
    // Eigen::Matrix<double, 1, 1> lbA(-1e10);
    // Eigen::Matrix<double, 1, 1> ubA(b_constraint);
    // 变量边界
    Eigen::Matrix<double, 7, 1> lb = -1e10 * Eigen::Matrix<double, 7, 1>::Ones();
    Eigen::Matrix<double, 7, 1> ub = 1e10 * Eigen::Matrix<double, 7, 1>::Ones();

    // 使用 qpOASES 求解 QP 问题
    QProblem qp(7, 1);
    Options options;
    options.printLevel = PL_LOW;
    qp.setOptions(options);
    int nWSR = 50;
    returnValue rval = qp.init(H_qp.data(), g_qp.data(), A_qp.data(), lb.data(), ub.data(),
                               lbA.data(), ubA.data(), nWSR);

    if (rval == SUCCESSFUL_RETURN) {
      Eigen::Matrix<double, 7, 1> tau_opt;
      qp.getPrimalSolution(tau_opt.data());

      tau_cmd = tau_opt;
  
  }


  //-------------------------------------------------------------------------------------------------------
  std_msgs::Float64 m;
  m.data = h0;
  h0_pub_.publish(m);
  if (h0_log_decim_ > 0 && (loop_counter_ % h0_log_decim_ == 0)) {
    t_log_.push_back((now - h0_log_t0_).toSec());
    h0_log_.push_back(h0);
  }

  double d = distance_ee_center.norm();  // 当前末端距离
  double margin = safeset_radius - d;    // 正：圈内 负：圈外

  // 将 h1 和 h3 保存到对应的数组中
  h0_values_.push_back(h0);
  h2_values_.push_back(h2);
  d_values_.push_back(d);
  eta_values_.push_back(pending_gp_data_[7]);

  // -------------------【HOCBF 功能集成 End】-------------------

  //==================================delay：根据trigger条件返回新的xy=================
  // 触发机制判断
  // if (has_pending_gp_) {
  //                             // 构造 x
  //   Eigen::VectorXd x(14), y(7);
  //   for (int i = 0; i < 7; ++i) {
  //     x(i) = robot_state.q[i];
  //     x(i + 7) = robot_state.dq[i];
  //     y(i) = tau_disturbance(i);  // 实时扰动
  //   }

  //   std_msgs::Float64MultiArray msg;
  //   msg.data.resize(21);
  //   for (int i = 0; i < 14; ++i)
  //     msg.data[i] = x(i);  // 接收到gp更新直接发送x

  //   if (trigger_met) {
  //     for (int i = 0; i < 7; ++i)
  //       msg.data[14 + i] = y(i);
  //     trigger_met = false;
  //     ROS_INFO("Controller: Trigger met, sent xy to GP Node.");
  //   } else {
  //     for (int i = 0; i < 7; ++i)
  //       msg.data[14 + i] = 9999.0;
  //     // ROS_INFO("Controller: No trigger, sent x + y=9999 to GP Node.");
  //   }
  //   gp_data_pub_.publish(msg);
  //   has_pending_gp_ = false;  // reset
  // }

  

  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(tau_cmd(i));
  }
  
  //===========================无delay恢复===============================
  // // publish joint x and disturbance y
  // std_msgs::Float64MultiArray msg;
  // msg.data.resize(7);
  // for (int i = 0; i < 7; ++i) {
  //   msg.data[i] = tau_disturbance(i);
  // }
  // debug_pub_.publish(msg);

  // // tau_cmd = saturateTorqueRate(tau_cmd, tau_J_d);

  // for (size_t i = 0; i < 7; ++i) {
  //   joint_handles_[i].setCommand(tau_cmd(i));
  // }
  //=======================================================================
}

// h0数据保存
void ForceExampleController::saveH0LogCsv() const {
  std::ostringstream oss;
  oss << h0_log_dir_ << "/" << h0_log_prefix_ << "_" << std::fixed << std::setprecision(0)
      << ros::Time::now().toSec() << ".csv";
  const std::string path = oss.str();

  std::ofstream ofs(path);
  if (!ofs) {
    ROS_ERROR_STREAM("[H0Log] Cannot open file: " << path);
    return;
  }
  ofs << "t_s,h0\n";
  const size_t N = std::min(h0_log_.size(), t_log_.size());
  ofs << std::setprecision(9);
  for (size_t i = 0; i < N; ++i) {
    ofs << t_log_[i] << "," << h0_log_[i] << "\n";
  }
  ofs.close();
  ROS_INFO_STREAM("[H0Log] Saved " << N << " samples to " << path);
}
// trigger vs. time 画图
ForceExampleController::~ForceExampleController() {
  std::ofstream ofs("/tmp/trigger_log.csv");
  for (auto step : trigger_steps_) {
    ofs << step << std::endl;
  }
  ofs.close();
  ROS_INFO_STREAM("Trigger log saved to /tmp/trigger_log.csv");
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::ForceExampleController,
                       controller_interface::ControllerBase)

// ---------------------------------查看tau的时间戳：（核对到毫秒级）
// 1）rostopic pub -r 10 /gp/tau_stamp std_msgs/Time "data: now"
// 2）rostopic pub -r 10 /gp/tau std_msgs/Float64MultiArray "data: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
// 0.7]" 分别以两个窗口发布 3）然后在控制器运行窗口会开始打印tau的发出以及接收时间
// 4）如果控制器窗口base delay很大：开一个新窗口写这个
// python - <<'PY'
// import rospy
// from std_msgs.msg import Time, Float64MultiArray
// rospy.init_node("fake_gp_pub")
// p_stamp = rospy.Publisher("/gp/tau_stamp", Time, queue_size=1)
// p_tau   = rospy.Publisher("/gp/tau", Float64MultiArray, queue_size=1)
// r = rospy.Rate(10)
// while not rospy.is_shutdown():
//     now = rospy.Time.now()
//     p_stamp.publish(Time(data=now))
//     p_tau.publish(Float64MultiArray(data=[0.1,0.2,0.3]))
//     r.sleep()
// PY
// 5）核对：rostopic echo -n 3 /gp/tau_stamp
// 确保三条每条都不同
// 6）这时控制器base delay会掉到毫秒级
// 结果：base delay大约0.2～1.9ms

//----------------------------------------如果启动franka之后一直是tau=0用下面的命令
// python - <<'PY'
// import rospy
// from sensor_msgs.msg import JointState
// rospy.init_node('gp_tau_test')
// pub = rospy.Publisher('/gp/tau', JointState, queue_size=10)
// rate = rospy.Rate(200)
// eff = [0.0]*7
// while not rospy.is_shutdown():
//     m = JointState(); m.header.stamp = rospy.Time.now(); m.effort = eff
//     pub.publish(m); rate.sleep()
// PY

// ------------------------------------------画图trigger time
// ./bashplot
//  python3 plot_trigger_log.py /tmp/trigger_log.csv --dt 0.001

//-----------------------------------------运行前打开窗口来保存h0数值
// python3 run_log_h0.py
//---------------------------------------------mc画图
// python3 h0_mc.py
