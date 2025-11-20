#ifndef GPR_H
#define GPR_H

#include <Eigen/Dense>
#include <tuple>
#include <utility>
#include <limits>

class GPR {
 public:
  // 构造函数声明
  GPR(int x_dim, int y_dim, int MaxDataQuantity, double SigmaN, double SigmaF,
      const Eigen::VectorXd& SigmaL, const Eigen::VectorXd& Lf_set, double compact_range);

  // 公共数据成员（如果不想暴露内部实现，可以改为私有并添加访问接口）
  Eigen::MatrixXd X, Y, K, L, aux_alpha, alpha;

  // 公共成员函数声明
  void addPoint(const Eigen::VectorXd x, const Eigen::VectorXd y);
  void deleteParam(int DeleteNr);
  std::tuple<Eigen::VectorXd, Eigen::MatrixXd, double, double, double, double>
  predict(const Eigen::VectorXd& x);
  void updateParam();
  std::pair<double, double> set_ErrorBound(const Eigen::VectorXd& x);
//offline gp----------------------------------------------
//void addOfflineRandomData(int num_samples, const Eigen::VectorXd& x_min, const Eigen::VectorXd& x_max);


//----------------------------------------------------------
 private:
  // 私有成员函数声明
  Eigen::MatrixXd kernel(const Eigen::MatrixXd& Xi, const Eigen::MatrixXd& Xj = Eigen::MatrixXd());

  // 私有数据成员
  int MaxDataQuantity, DataQuantity, x_dim, y_dim;
  Eigen::VectorXd xMax, xMin;
  double SigmaN, SigmaF, compact_range, Lf, Lk, tau, delta, LsigmaN2, LmuN, M;
  Eigen::VectorXd SigmaL, Lf_set;
};

#endif  // GPR_H
