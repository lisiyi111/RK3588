/**
  *************************************************
  * @file               :utils.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/6                
  *************************************************
  */

#include "stdeploy/utils/log_util.h"
#include "stdeploy/utils/md5.hpp"

#include <sstream>


namespace stdeploy {

    int64_t GetCurrentTimeUs() {
        struct timeval tv{};
        gettimeofday(&tv, NULL);
        return tv.tv_sec * 1000000 + tv.tv_usec;
    }

    int GetLibVersion(std::string &lib_version_str) {
        lib_version_str = STDEPLOY_VERSION;
        return 0;
    }

    int GetFileMd5sum(std::string &file_path, std::string &md5sum_str) {
        MD5 *md5_server = new MD5();
        ifstream ifs(file_path, std::ios::in);
        if (!ifs) {
            STDEPLOY_ERROR("read model_path error: %s", file_path.c_str());
            return -1;
        }
        md5_server->update(ifs);
        md5sum_str = md5_server->toString();
        ifs.close();
        delete md5_server;
        return 0;
    }

    bool SDLogger::enable_info = true;
    bool SDLogger::enable_warning = true;

    void SetLogger(bool enable_info, bool enable_warning) {
        SDLogger::enable_info = enable_info;
        SDLogger::enable_warning = enable_warning;
    }

    SDLogger::SDLogger(bool verbose, const std::string &prefix) {
        verbose_ = verbose;
        line_ = "";
#ifdef ENABLE_ANDROID
        prefix_ = std::string("stdeploy-ai");
#else
        prefix_ = prefix;
#endif
    }

    SDLogger &SDLogger::operator<<(std::ostream &(*os)(std::ostream &)) {
        if (!verbose_) {
            return *this;
        }
        std::cout << prefix_ << " " << line_ << std::endl;
#ifdef ENABLE_ANDROID
        __android_log_print(ANDROID_LOG_INFO, prefix_.c_str(), "%s", line_.c_str());
#endif
        line_ = "";
        return *this;
    }

} //namespace stdeploy