#include "GPR.h"
#include <cmath>
#include <iostream>
#include <ros/ros.h>
#include <fstream>
//offline gp
#include <random>



// 构造函数实现
GPR::GPR(int x_dim, int y_dim, int MaxDataQuantity, double SigmaN, double SigmaF,
         const Eigen::VectorXd& SigmaL, const Eigen::VectorXd& Lf_set, double compact_range)
    : MaxDataQuantity(MaxDataQuantity), x_dim(x_dim), y_dim(y_dim),
      SigmaN(SigmaN), SigmaF(SigmaF), SigmaL(SigmaL), Lf_set(Lf_set), compact_range(compact_range) {
  DataQuantity = 0;
  X = Eigen::MatrixXd::Constant(x_dim, MaxDataQuantity, std::numeric_limits<double>::quiet_NaN());
  Y = Eigen::MatrixXd::Constant(MaxDataQuantity, y_dim, std::numeric_limits<double>::quiet_NaN());
  K = Eigen::MatrixXd::Constant(MaxDataQuantity, MaxDataQuantity, std::numeric_limits<double>::quiet_NaN());
  L = Eigen::MatrixXd::Constant(MaxDataQuantity, MaxDataQuantity, std::numeric_limits<double>::quiet_NaN());
  aux_alpha = Eigen::MatrixXd::Constant(MaxDataQuantity, y_dim, std::numeric_limits<double>::quiet_NaN());
  alpha = Eigen::MatrixXd::Constant(MaxDataQuantity, y_dim, std::numeric_limits<double>::quiet_NaN());
  xMax = Eigen::VectorXd::Constant(x_dim, std::numeric_limits<double>::quiet_NaN());
  xMin = Eigen::VectorXd::Constant(x_dim, std::numeric_limits<double>::quiet_NaN());
  tau = 1e-10;
  delta = 0.01;
  Lf = sqrt(Lf_set.squaredNorm());
  Lk = SigmaF * SigmaF * exp(-0.5) / SigmaL.norm();
}

// kernel 函数实现
Eigen::MatrixXd GPR::kernel(const Eigen::MatrixXd& Xi, const Eigen::MatrixXd& Xj) {
  int n = Xi.cols();
  int m = (Xj.size() == 0) ? n : Xj.cols();
  Eigen::MatrixXd kern = Eigen::MatrixXd::Zero(n, m);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      Eigen::VectorXd Xj_col = (Xj.size() == 0) ? Xi.col(j) : Xj.col(j);
      Eigen::VectorXd diff = Xi.col(i) - Xj_col;
      kern(i, j) = (SigmaF * SigmaF) * exp(-0.5 * (diff.array().square() / SigmaL.array().square()).sum());
    }
  }
  return kern;
}

// addPoint 函数实现
void GPR::addPoint(const Eigen::VectorXd x, const Eigen::VectorXd y) {
  if (DataQuantity >= MaxDataQuantity) {
    deleteParam(1);
    addPoint(x, y);
  } else {
    X.block(0, DataQuantity, x_dim, 1) = x;
    Y.block(DataQuantity, 0, 1, y_dim) = y.transpose();
    DataQuantity = DataQuantity + 1;
    updateParam();
    for (int i = 0; i < x_dim; ++i) {
      xMax(i) = X.row(i).head(DataQuantity).maxCoeff();
      xMin(i) = X.row(i).head(DataQuantity).minCoeff();
    }

    // 调试信息：输出当前的数据对数量和最新添加的 x 与 y
    ROS_INFO_STREAM("Added new data point. Current DataQuantity: " << DataQuantity);
    // ROS_INFO_STREAM("New x: " << x.transpose());
    // ROS_INFO_STREAM("New y: " << y.transpose());

  }
}

// updateParam 函数实现
void GPR::updateParam() {
  int LocalGP_DataQuantity = DataQuantity;
  Eigen::MatrixXd X_set = X.block(0, 0, x_dim, LocalGP_DataQuantity);
  if (DataQuantity == 1) {
    Eigen::MatrixXd new_K = kernel(X_set, X_set) + Eigen::MatrixXd::Constant(1, 1, SigmaN * SigmaN);
    Eigen::MatrixXd new_L = new_K.llt().matrixL();
    K.block(0, 0, LocalGP_DataQuantity, LocalGP_DataQuantity) = new_K;
    L.block(0, 0, LocalGP_DataQuantity, LocalGP_DataQuantity) = new_L;
    for (int i = 0; i < y_dim; ++i) {
      Eigen::VectorXd y_set = Y.block(0, i, LocalGP_DataQuantity, 1);
      Eigen::VectorXd new_aux_alpha = new_L.triangularView<Eigen::Lower>().solve(y_set);
      Eigen::VectorXd new_alpha = new_L.transpose().triangularView<Eigen::Upper>().solve(new_aux_alpha);
      aux_alpha.block(0, i, LocalGP_DataQuantity, 1) = new_aux_alpha;
      alpha.block(0, i, LocalGP_DataQuantity, 1) = new_alpha;
    }
  } else {
    Eigen::MatrixXd old_X = X_set.block(0, 0, x_dim, LocalGP_DataQuantity - 1);
    Eigen::VectorXd new_x = X_set.col(LocalGP_DataQuantity - 1);
    Eigen::MatrixXd old_K = K.block(0, 0, LocalGP_DataQuantity - 1, LocalGP_DataQuantity - 1);
    Eigen::MatrixXd old_L = L.block(0, 0, LocalGP_DataQuantity - 1, LocalGP_DataQuantity - 1);
    Eigen::MatrixXd b = kernel(old_X, new_x);
    double c = kernel(new_x, new_x)(0, 0) + SigmaN * SigmaN;
    Eigen::MatrixXd L1 = old_L.triangularView<Eigen::Lower>().solve(b);
    double L2 = sqrt(c - L1.norm() * L1.norm());
    Eigen::MatrixXd new_K(LocalGP_DataQuantity, LocalGP_DataQuantity);
    new_K << old_K, b, b.transpose(), c;
    Eigen::MatrixXd new_L = Eigen::MatrixXd::Zero(LocalGP_DataQuantity, LocalGP_DataQuantity);
    new_L.block(0, 0, LocalGP_DataQuantity - 1, LocalGP_DataQuantity - 1) = old_L;
    new_L.block(LocalGP_DataQuantity - 1, 0, 1, LocalGP_DataQuantity - 1) = L1.transpose();
    new_L(LocalGP_DataQuantity - 1, LocalGP_DataQuantity - 1) = L2;
    K.block(0, 0, LocalGP_DataQuantity, LocalGP_DataQuantity) = new_K;
    L.block(0, 0, LocalGP_DataQuantity, LocalGP_DataQuantity) = new_L;
    for (int i = 0; i < y_dim; ++i) {
      Eigen::MatrixXd old_aux_alpha = aux_alpha.block(0, i, LocalGP_DataQuantity - 1, 1);
      double new_y = Y(LocalGP_DataQuantity - 1, i);
      double matrixProduct = (L1.transpose() * old_aux_alpha)(0, 0);
      double new_aux_alpha = (new_y - matrixProduct) / L2;
      aux_alpha(LocalGP_DataQuantity - 1, i) = new_aux_alpha;
      Eigen::MatrixXd new_alpha = new_L.transpose().triangularView<Eigen::Upper>().solve(aux_alpha.block(0, i, LocalGP_DataQuantity, 1));
      alpha.block(0, i, LocalGP_DataQuantity, 1) = new_alpha;
    }
  }
}

// deleteParam 函数实现
void GPR::deleteParam(int DeleteNr) {
  Eigen::MatrixXd old_K = K.topLeftCorner(DataQuantity, DataQuantity);
  Eigen::MatrixXd old_L = L.topLeftCorner(DataQuantity, DataQuantity);
  Eigen::MatrixXd Lb = old_L.block(DeleteNr, DeleteNr, DataQuantity - DeleteNr, DataQuantity - DeleteNr);
  Eigen::MatrixXd lbc = old_L.block(DeleteNr, DeleteNr - 1, DataQuantity - DeleteNr, 1);
  Eigen::MatrixXd A0 = Lb * Lb.transpose();
  Eigen::MatrixXd A1 = A0 + lbc * lbc.transpose();
  Eigen::MatrixXd Lb_star = A1.llt().matrixL();
  K.topLeftCorner(DataQuantity - 1, DataQuantity - 1) <<
      old_K.topLeftCorner(DeleteNr - 1, DeleteNr - 1), old_K.block(0, DeleteNr, DeleteNr - 1, DataQuantity - DeleteNr),
      old_K.block(DeleteNr, 0, DataQuantity - DeleteNr, DeleteNr - 1), old_K.block(DeleteNr, DeleteNr, DataQuantity - DeleteNr, DataQuantity - DeleteNr);
  L.topLeftCorner(DataQuantity - 1, DataQuantity - 1) <<
      old_L.topLeftCorner(DeleteNr - 1, DeleteNr - 1), Eigen::MatrixXd::Zero(DeleteNr - 1, DataQuantity - DeleteNr),
      old_L.block(DeleteNr, 0, DataQuantity - DeleteNr, DeleteNr - 1), Lb_star;
  for (int i = 0; i < y_dim; ++i) {
    Eigen::VectorXd old_aux_alpha = aux_alpha.col(i).head(DataQuantity);
    Eigen::VectorXd aa = old_aux_alpha.head(DeleteNr - 1);
    Eigen::VectorXd ab = old_aux_alpha.segment(DeleteNr, DataQuantity - DeleteNr);
    double ac = old_aux_alpha(DeleteNr - 1);
    aux_alpha.col(i).head(DataQuantity - 1) << aa, Lb_star.triangularView<Eigen::Lower>().solve(lbc * ac + Lb * ab);
    alpha.col(i).head(DataQuantity - 1) = L.topLeftCorner(DataQuantity - 1, DataQuantity - 1).triangularView<Eigen::Lower>().solve(aux_alpha.col(i).head(DataQuantity - 1));
  }
  DataQuantity = DataQuantity - 1;
  X.block(0, DeleteNr - 1, X.rows(), DataQuantity - DeleteNr + 1) = X.block(0, DeleteNr, X.rows(), DataQuantity - DeleteNr + 1);
  Y.block(DeleteNr - 1, 0, DataQuantity - DeleteNr + 1, Y.cols()) = Y.block(DeleteNr, 0, DataQuantity - DeleteNr + 1, Y.cols());
}

// set_ErrorBound 函数实现
std::pair<double, double> GPR::set_ErrorBound(const Eigen::VectorXd& x) {
  int LocalGP_DataQuantity = DataQuantity;
  M = std::pow(compact_range, x_dim) * std::sqrt(x_dim) / (2 * tau);
  Eigen::MatrixXd All_K = K.block(0, 0, LocalGP_DataQuantity, LocalGP_DataQuantity);
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(All_K, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::MatrixXd S_inv = svd.singularValues().asDiagonal().inverse();
  Eigen::MatrixXd A11K_inv = svd.matrixV() * S_inv * svd.matrixU().transpose();
  LsigmaN2 = 2 * Lk * tau * (1 + LocalGP_DataQuantity * A11K_inv.norm() * SigmaF * SigmaF);
  Eigen::VectorXd Lmu_set(y_dim);
  for (int i = 0; i < y_dim; ++i) {
    Eigen::VectorXd alpha_i = alpha.block(0, i, LocalGP_DataQuantity, 1);
    Lmu_set(i) = Lk * std::sqrt(LocalGP_DataQuantity) * alpha_i.norm();
  }
  double LmuN = Lmu_set.norm();
   double beta = 2 * std::log(M / delta);
  // double beta = 0.5;
  double gamma = (Lf + LmuN) * tau + std::sqrt(beta * LsigmaN2 * tau);
  return std::make_pair(beta, gamma);
}

// predict 函数实现
std::tuple<Eigen::VectorXd, Eigen::MatrixXd, double, double, double, double>
GPR::predict(const Eigen::VectorXd& x) {
  int LocalGP_DataQuantity = DataQuantity;
  Eigen::VectorXd mu(y_dim);
  Eigen::MatrixXd var(y_dim, y_dim);
  double beta, gamma, eta, eta_min;
  if (LocalGP_DataQuantity == 0) {
    mu = Eigen::VectorXd::Zero(y_dim);
    var = Eigen::MatrixXd::Constant(y_dim, y_dim, SigmaF * SigmaF);
  } else {
    Eigen::MatrixXd X_set = X.block(0, 0, x_dim, LocalGP_DataQuantity);
    Eigen::MatrixXd temp_L = L.block(0, 0, LocalGP_DataQuantity, LocalGP_DataQuantity);
    Eigen::VectorXd v = temp_L.triangularView<Eigen::Lower>().solve(kernel(X_set, x));
    Eigen::MatrixXd k_xx = kernel(x);
    double var_value = (k_xx - v.transpose() * v)(0, 0);
    Eigen::VectorXd var_vector = Eigen::VectorXd::Constant(y_dim, var_value);
    var = var_vector.asDiagonal();
    mu = aux_alpha.block(0, 0, LocalGP_DataQuantity, y_dim).transpose() * v;
    std::tie(beta, gamma) = set_ErrorBound(x);
    eta = sqrt(beta) * sqrt(var.maxCoeff()) + gamma;
    eta_min = sqrt(beta) * SigmaN + gamma;
  }

    // 输出预测结果
//   ROS_INFO_STREAM("GP Prediction: mu = " << mu.transpose());
//   ROS_INFO_STREAM("GP Prediction: var = " << var);
//   ROS_INFO_STREAM("GP Prediction: beta = " << beta << ", gamma = " << gamma);
//   ROS_INFO_STREAM("GP Prediction: eta = " << eta << ", eta_min = " << eta_min);

  return std::make_tuple(mu, var, beta, gamma, eta, eta_min);
}

 //offline gp=================================
//  void GPR::addOfflineRandomData(int num_samples, const Eigen::VectorXd& x_min, const Eigen::VectorXd& x_max) {
//   std::random_device rd;
//   std::mt19937 gen(rd());

//   for (int i = 0; i < num_samples; ++i) {
//     Eigen::VectorXd x_sample(x_dim);
//     for (int j = 0; j < x_dim; ++j) {
//       std::uniform_real_distribution<> dis(x_min(j), x_max(j));
//       x_sample(j) = dis(gen);
//     }

//     Eigen::VectorXd y_sample(y_dim);
//     for (int k = 0; k < y_dim; ++k) {
//       // 你可以换成更合理的扰动模型
//       y_sample(k) = 5.0 * std::sin(3.0 * x_sample(k % x_dim));  // 只是个示例函数
//     }

//     this->addPoint(x_sample, y_sample);
//   }

//   ROS_INFO_STREAM("[Offline GP] Generated " << num_samples << " random samples.");
// }
//==============================================================