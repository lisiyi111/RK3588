/**
  *************************************************
  * @file               :utils.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/6                
  *************************************************
  */

#pragma once

#include <cstdio>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <cassert>
#include <sys/time.h>


#if defined(_WIN32)
#ifdef STDEPLOY_LIB
#define STDEPLOY_DECL __declspec(dllexport)
#else
#define STDEPLOY_DECL __declspec(dllimport)
#endif  // _WIN32
#else
#define STDEPLOY_DECL __attribute__((visibility("default")))
#endif  // _LINUX


#ifdef ENABLE_ANDROID
#include <android/log.h>  // NOLINT
#endif


#ifdef _WIN32
#define OS_PATH_SEP "\\"
#else
#define OS_PATH_SEP "/"
#endif

#include "stdeploy/version/version.h"

namespace stdeploy {

    inline static std::string extractFileName(const char *fullPath) {
        std::string pathStr(fullPath);
        size_t found = pathStr.find_last_of(OS_PATH_SEP); // 查找最后一个目录分隔符
        return pathStr.substr(found + 1); // 提取文件名部分
    }

    STDEPLOY_DECL int64_t GetCurrentTimeUs();

    STDEPLOY_DECL int GetLibVersion(std::string &lib_version_str);

    STDEPLOY_DECL int GetFileMd5sum(std::string &file_path, std::string &md5sum_str);

    class STDEPLOY_DECL SDLogger {
    public:
        static bool enable_info;
        static bool enable_warning;

        SDLogger() {
            line_ = "";
            prefix_ = "[StDeploy]";
            verbose_ = true;
        }

        // explicit 用在类构造函数中，限制隐式类型转换
        explicit SDLogger(bool verbose, const std::string &prefix = "[StDeploy]");

        template<typename T>
        SDLogger &operator<<(const T &val) {
            if (!verbose_) {
                return *this;
            }
            std::stringstream ss;
            ss << val;
            line_ += ss.str();
            return *this;
        }

        SDLogger &operator<<(std::ostream &(*os)(std::ostream &));

        ~SDLogger() {
            if (verbose_ && line_ != "") {
                std::cout << line_ << std::endl;
#ifdef ENABLE_ANDROID
                __android_log_print(ANDROID_LOG_INFO, prefix_.c_str(), "%s",
                          line_.c_str());
#endif
            }
        }

    private:
        std::string line_;
        std::string prefix_;
        bool verbose_ = true;
    };

    void SetLogger(bool enable_info = true, bool enable_warning = true);

    /// 函数模板,Str打印 std::vector
    template<typename T>
    std::string Str(const std::vector<T> &shape) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < shape.size(); i++) {
            oss << " " << shape[i];
        }
        oss << "]";
        return oss.str();
    }

}// namespace stdeploy


#ifndef __REL_FILE__
#define __REL_FILE__ stdeploy::extractFileName(__FILE__)
//#define __REL_FILE__ __FILE__

#endif

#define SDERROR                                                                \
  SDLogger(true, "[ERROR]")                                                    \
      << __REL_FILE__ << "(" << __LINE__ << ")::" << __FUNCTION__ << "\t"

#define SDWARNING                                                              \
  SDLogger(stdeploy::SDLogger::enable_warning, "[WARN]")                    \
      << __REL_FILE__ << "(" << __LINE__ << ")::" << __FUNCTION__ << "\t"

#define SDINFO                                                                 \
  SDLogger(stdeploy::SDLogger::enable_info, "[INFO]")                          \
                           << __REL_FILE__ << "(" << __LINE__                  \
                           << ")::" << __FUNCTION__ << "\t"


#define SDASSERT(condition, format, ...)                                       \
  if (!(condition)) {                                                          \
    int n = std::snprintf(nullptr, 0, format, ##__VA_ARGS__);                  \
    std::vector<char> out(n + 1);                                           \
    std::snprintf(out.data(), n + 1, format, ##__VA_ARGS__);                \
    stdeploy::SDERROR << out.data() << std::endl;                                     \
    std::abort();                                                              \
  }

#define STDEPLOY_INFO(format, ...)                                             \
    {                                                                          \
        int n = std::snprintf(nullptr, 0, format, ##__VA_ARGS__);              \
        std::vector<char> out(n + 1);                                       \
        std::snprintf(out.data(), n + 1, format, ##__VA_ARGS__);            \
        stdeploy::SDINFO << out.data() << std::endl;                                  \
    }                                                                                                                                   \


#define STDEPLOY_ERROR(format, ...)                                            \
  {                                                                            \
    int n = std::snprintf(nullptr, 0, format, ##__VA_ARGS__);                  \
    std::vector<char> out(n + 1);                                           \
    std::snprintf(out.data(), n + 1, format, ##__VA_ARGS__);                \
    stdeploy::SDERROR << out.data() << std::endl;                                     \
  }


#define STDEPLOY_WARNING(format, ...)                                          \
  {                                                                            \
    int n = std::snprintf(nullptr, 0, format, ##__VA_ARGS__);                  \
    std::vector<char> out(n + 1);                                           \
    std::snprintf(out.data(), n + 1, format, ##__VA_ARGS__);                \
    stdeploy::SDWARNING << out.data() << std::endl;                                   \
  }

