// gp_node.cpp
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <Eigen/Dense>
#include "GPR.h"
// delay
#include <deque>
#include <std_msgs/Bool.h>
#include <utility>

using namespace Eigen;

class GPNode
{
public:
  GPNode(ros::NodeHandle &nh) : nh_(nh)
  {
    // 初始化 GP 模型参数
    int x_dim = 14; // 7位置 + 7速度 + 7期望加速度(position and velocity is enough , acceleration is set to zero now)
    int y_dim = 7;  // 输出7维扰动/补偿
    int MaxDataQuantity = 100;
    double SigmaN = 0.01;                    // 0.01
    double SigmaF = 1;                       // 1
    VectorXd SigmaL = VectorXd::Ones(x_dim); // 1*
    VectorXd Lf_set = VectorXd::Zero(x_dim);
    double compact_range = 10.0;
    gpr_ = new GPR(x_dim, y_dim, MaxDataQuantity, SigmaN, SigmaF, SigmaL, Lf_set, compact_range);
    // offline gp============================
    //  VectorXd x_min = VectorXd::Constant(x_dim, -2.0);  // 根据你任务修改
    //  VectorXd x_max = VectorXd::Constant(x_dim,  2.0);

    // int num_samples = 850;  // 离线数据样本数量
    // gpr_->addOfflineRandomData(num_samples, x_min, x_max);
    //=================================

    //========================online恢复==============
    // 发布 GP 补偿结果到 /gp_compensation
    gp_comp_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("/gp_compensation", 10);
    // 订阅 controller 发来的 GP 训练数据
    gp_data_sub_ = nh_.subscribe("/gp_training_data", 1, &GPNode::gpDataCallback, this);

    //=====================delay：接收controller发来的xy===============================

    check_timer_ = nh_.createTimer(ros::Duration(0.001), &GPNode::checkAndPublishDelayedMsg, this);

    delay_seconds_ = 0.015; // 10ms 延迟

    //========================online恢复结束========================
  }

  ~GPNode()
  {
    if (gpr_ != nullptr)
    {
      delete gpr_;
    }
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber joint_state_sub_;
  ros::Subscriber debug_tau_sub_;
  ros::Subscriber gp_data_sub_; // gpnode接收k=0时的xy
  ros::Publisher gp_comp_pub_;
  GPR *gpr_;
  //======================delay：接收controller发来的xy=================
  std::deque<std::pair<ros::Time, std_msgs::Float64MultiArray>> delay_queue_;
  ros::Timer check_timer_;
  double delay_seconds_; // 延迟时间
  //=================================================================

  // 用于缓存最新的状态和力矩数据
  // joint_state
  Eigen::Matrix<double, 7, 1> current_q_;
  Eigen::Matrix<double, 7, 1> current_dq_;
  // debug_tau
  Eigen::Matrix<double, 7, 1> current_tau_pd_;
  Eigen::Matrix<double, 7, 1> current_tau_cmd_;
  Eigen::Matrix<double, 7, 1> current_tau_disturbance_;

  // 是否收到有效数据
  bool joint_state_received_;
  bool tau_received_;

private:
  //========================delay：接收controller发来的xy （k=0）=============
  void gpDataCallback(const std_msgs::Float64MultiArray::ConstPtr &msg)
  {
    if (msg->data.size() != 21)
    {
      ROS_ERROR("GP Node: Expected 21-dimensional [x(14)+y(7)] message.");
      return;
    }
    Eigen::VectorXd x(14), y(7);
    for (int i = 0; i < 14; ++i)
      x(i) = msg->data[i];
    for (int i = 0; i < 7; ++i)
      y(i) = msg->data[14 + i];

    // 调用你已有的统一处理接口
    current_q_ = x.head<7>();
    current_dq_ = x.tail<7>();
    current_tau_disturbance_ = y;

    joint_state_received_ = true;
    tau_received_ = true;

    updateGP();
  }

  //===========================delay：延迟发布函数==========================
  void checkAndPublishDelayedMsg(const ros::TimerEvent &)
  {
    if (delay_queue_.empty())
      return;

    ros::Time now = ros::Time::now();

    // while (!delay_queue_.empty())
    // {
    //   auto &front = delay_queue_.front();
    //   ros::Duration diff = now - front.first;

    //   if (diff.toSec() >= delay_seconds_)
    //   {
    //     gp_comp_pub_.publish(front.second);

    //     ROS_INFO_STREAM("[GP Node] Published delayed msg "
    //       << "(delay=" << diff.toSec() << " s, eta=" << front.second.data[7]
    //       << ") at " << now);
    //     // ROS_INFO_STREAM("[GP Node] Published delayed msg (eta = " << front.second.data[7]
    //     //                                                           << ") at " << now);

    //     delay_queue_.pop_front();
     
    //     }else
    //   {
    //     break; // 队列头还没到时间
    //   }
    // }
    while (!delay_queue_.empty() && now >= delay_queue_.front().first)
    {
      const auto &front = delay_queue_.front();
      gp_comp_pub_.publish(front.second);

      double true_delay = (now - (front.first - ros::Duration(delay_seconds_))).toSec();
      ROS_INFO_STREAM("[GP Node] Published delayed msg (eta = "
                      << front.second.data[7]
                      << ", true delay = " << true_delay << " s) at " << now);

      delay_queue_.pop_front();
    }
  }
  //=================================================================

  // 当 joint_states 和 debug_tau 都有新数据后，调用此函数
  void updateGP()
  {

    // 构造 x: 21维  [7位置 + 7速度 + 7期望加速度(先写0)]
    VectorXd x(14);
    for (int i = 0; i < 7; ++i)
    {
      x(i) = current_q_(i);
    }
    for (int i = 0; i < 7; ++i)
    {
      x(7 + i) = current_dq_(i);
    }

    // 计算 y：例如 y = tau_cmd - tau_pd
    //  代表“相对于 PD 的差值”

    // 转成 VectorXd
    VectorXd y(7);
    for (int i = 0; i < 7; ++i)
    {
      y(i) = current_tau_disturbance_(i);
    }
    //==============================delay: 接收controller发来的xy（k>0)并判断是否更新==============
    // 检查 y 是否有效
    bool valid_y = true;
    for (int i = 0; i < 7; ++i)
    {
      if (std::abs(y(i)) >= 9999.0)
      {
        valid_y = false;
        break;
      }
    }

    // 如果是无效 y（即仅 x），什么也不做
    if (!valid_y)
    {
      ROS_DEBUG("[GP Node] Received x-only sample (y=9999). Skip GP update.");
      return;
    }
    ROS_INFO("[GP Node] Received valid (x, y). Performing GP update...");

    //只有xy都收到时才预测更新
    gpr_->addPoint(x, y);
    ROS_INFO("[GP Node] addPoint(x, y) done.");
    //=======================================================================================

    // 调用 predict
    auto predict_result = gpr_->predict(x);
    VectorXd mu = std::get<0>(predict_result); // 7维补偿
    double eta = std::get<4>(predict_result);  // 误差界限

    // 打印eta和mu
    // ROS_INFO_STREAM("[GP Node] Predict eta = " << eta << ", mu = " << mu.transpose());

    // 发布到 /gp_compensation
    std_msgs::Float64MultiArray comp_msg;
    comp_msg.data.resize(8);
    for (int i = 0; i < 7; ++i)
    {
      comp_msg.data[i] = mu(i);
    }
    comp_msg.data[7] = eta;
    ROS_INFO_STREAM("Predict at: " << ros::Time::now());

    //========================delay：延迟发布eta，mu===========================
    // delay_queue_.push_back({ros::Time::now(), comp_msg});
    delay_queue_.push_back({ros::Time::now() + ros::Duration(delay_seconds_), comp_msg});

    //===============================================================

    // gp_comp_pub_.publish(comp_msg); /无delay实时发布

    // //打印调试信息
    // ROS_INFO_STREAM("GP update done. x=" << x.transpose()
    //                 << ", y=" << y.transpose()) ;
    //                 << ", predicted mu=" << mu.transpose()
    //                 << ", error bound eta=" << eta);

    ROS_INFO_STREAM("error bound eta=" << comp_msg.data[7]);
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "gp_node");
  ros::NodeHandle nh;
  GPNode gp_node(nh);
  ros::spin(); // 需要恢复
  return 0;
}
