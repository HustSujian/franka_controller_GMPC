#pragma once
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>
#include <vector>

namespace serl_franka_controllers {

class CSVLogger {
public:
  CSVLogger() = default;
  ~CSVLogger() { close(); }

  // 打开文件 + 写表头（只做一次）
  bool open(const std::string& filename, const std::string& header) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (ofs_.is_open()) return true;
    ofs_.open(filename, std::ios::out | std::ios::trunc);
    if (!ofs_.is_open()) return false;
    ofs_ << std::fixed << std::setprecision(6);
    ofs_ << header << "\n";
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (ofs_.is_open()) ofs_.close();
  }

  // 记录：t + 12维 p0
  void log_p0(double t, const Eigen::Matrix<double,12,1>& p0) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!ofs_.is_open()) return;
    ofs_ << t;
    for (int i = 0; i < 12; ++i) ofs_ << "," << p0(i);
    ofs_ << "\n";
  }

  // 记录：t + p0 + Vd + tau
  void log_full(double t,
                const Eigen::Matrix<double,12,1>& p0,
                const Eigen::Matrix<double,6,1>& Vd,
                const Eigen::Matrix<double,7,1>& tau) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!ofs_.is_open()) return;
    ofs_ << t;
    for (int i = 0; i < 12; ++i) ofs_ << "," << p0(i);
    for (int i = 0; i < 6;  ++i) ofs_ << "," << Vd(i);
    for (int i = 0; i < 7;  ++i) ofs_ << "," << tau(i);
    ofs_ << "\n";
  }

  void log_primary_final(double t,
                       const Eigen::Matrix<double,12,1>& p0,
                       const Eigen::Matrix<double,6,1>& Vd,
                       const Eigen::Matrix<double,7,1>& u_primary,
                       const Eigen::Matrix<double,7,1>& u_final)
    {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!ofs_.is_open()) return;

    ofs_ << t;
    for (int i = 0; i < 12; ++i) ofs_ << "," << p0(i);
    for (int i = 0; i < 6;  ++i) ofs_ << "," << Vd(i);
    for (int i = 0; i < 7;  ++i) ofs_ << "," << u_primary(i);
    for (int i = 0; i < 7;  ++i) ofs_ << "," << u_final(i);
    ofs_ << "\n";
    ofs_.flush();
    }


private:
  std::ofstream ofs_;
  std::mutex mtx_;
};

}  // namespace serl_franka_controllers
