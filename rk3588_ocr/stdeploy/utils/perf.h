/**
  *************************************************
  * @file               :perf.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/5/8                
  *************************************************
  */

#pragma once

#include "stdeploy/utils/log_util.h"
#include <chrono> // NOLINT

namespace stdeploy {

    class STDEPLOY_DECL TimeCount {
    public:
        void Start() { begin_ = std::chrono::system_clock::now(); }

        void End() { end_ = std::chrono::system_clock::now(); }

        double Duration() {
            auto duration =
                    std::chrono::duration_cast<std::chrono::microseconds>(end_ - begin_);

            return static_cast<double>(duration.count()) / 1000.;
        };

        void PrintInfo(const std::string &prefix = "TimeCounter: ",
                       bool print_out = true) {
            if (!print_out) {
                return;
            }
            SDLogger() << prefix << " duration = " << Duration() << "ms." << std::endl;
        }

    private:
        std::chrono::time_point<std::chrono::system_clock> begin_;
        std::chrono::time_point<std::chrono::system_clock> end_;

    };

} //namespace stdeploy