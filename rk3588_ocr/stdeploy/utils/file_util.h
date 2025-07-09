/**
  *************************************************
  * @file               :file_util.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/12                
  *************************************************
  */

#pragma once

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <unistd.h>
#include <vector>
#include <string>

#include "stdeploy/utils/log_util.h"

namespace stdeploy {
    namespace utils {
        /**
         * @brief Read data from file
         *
         * @param path [in] File path
         * @param out_data [out] Read data
         * @return int -1: error; > 0: Read data size
         */
        STDEPLOY_DECL int read_data_from_file(const char *path, char **out_data);

        // load float list from bin
        STDEPLOY_DECL std::vector<float> loadFloatArrayFromBin(const std::string &filename, size_t size);

        // save float list to bin
        STDEPLOY_DECL void saveFloatArrayToBin(const float *data, size_t size, const std::string &filename);

        // replace file suffix
        STDEPLOY_DECL std::string replace_file_ext(const std::string &file_path, std::string &new_ext_name);

        // read bin file to string
        STDEPLOY_DECL bool ReadBinaryFromFile(const std::string &file, std::string *contents);

        // os.path.join
        STDEPLOY_DECL std::string
        path_join(const std::vector<std::string> &paths, const std::string &sep = OS_PATH_SEP);

        // os.path.join
        STDEPLOY_DECL std::string path_join(const std::string &folder, const std::string &filename);

        // os.path.basename
        STDEPLOY_DECL std::string get_basename(const std::string &path);

        // os.path.splitext(basename)[0]
        STDEPLOY_DECL std::string get_basename_without_suffix(const std::string &path);

        // os.path.splitext(basename)[1]
        STDEPLOY_DECL std::string get_file_suffix(const std::string &path);

        // os.path.dirname
        STDEPLOY_DECL std::string get_parentDir_name(const std::string &path);

        // os.path.dirname sub to get dir name
        STDEPLOY_DECL std::string get_subDir_name(const std::string &path);

        // judge path exist
        STDEPLOY_DECL bool path_exists(const std::string &path);

        // mkdir
        STDEPLOY_DECL void make_dir(const std::string &path);

        // mkdirs
        STDEPLOY_DECL void make_dirs(const std::string &path);

        // split string
        STDEPLOY_DECL std::vector<std::string> split_string(const std::string &content, const std::string &delimiter);

        // judge file exist
        STDEPLOY_DECL bool file_exist(const std::string &path);

        // save file to std::vector<char>
        STDEPLOY_DECL bool dump_file(const std::string &path, std::vector<char> &data);

        // save file to char*
        STDEPLOY_DECL bool dump_file(const std::string &path, char *data, int size);

        // read file to std::vector<char>
        STDEPLOY_DECL bool read_file(const std::string &path, std::vector<char> &data);

        // judge path is dir
        STDEPLOY_DECL bool is_dir(const std::string &path);

        /**
         * judge value is in vector
         */
        template<typename T>
        bool is_contains(const std::vector<T> &vec, const T &value) {
            return std::find(vec.begin(), vec.end(), value) != vec.end();
        }

        // cmp filename by number 1/2
        STDEPLOY_DECL bool cmp_filenames_by_number(const std::string &a, const std::string &b);

        // load img file
        STDEPLOY_DECL bool load_img_files(const std::string &path, std::vector<std::string> &img_files);

        STDEPLOY_DECL bool load_img_files_sort_by_number(const std::string &path, std::vector<std::string> &img_files);

        // judge the file is end with img suffix
        STDEPLOY_DECL bool is_supported_image_extension(const std::string &filename);

        // read buffer from .yuv
        STDEPLOY_DECL int read_yuv_data(const std::string &filename, unsigned char *data, int size);

        // save buffer to .yuv
        STDEPLOY_DECL int save_yuv_data(const std::string &filename, const unsigned char *data, int size);

        // read txt to vector
        STDEPLOY_DECL std::vector<std::string> read_txt_to_vector(const std::string &filename);

    } //namespace utils
} //namespace stdeploy


