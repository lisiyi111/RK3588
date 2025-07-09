/**
  *************************************************
  * @file               :file_util.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/12                
  *************************************************
  */

#include "stdeploy/utils/file_util.h"

namespace stdeploy {
    namespace utils {

        std::vector<std::string> read_txt_to_vector(const std::string &filename){
            std::vector<std::string> lines;  // 存储每一行的字符串
            std::ifstream file(filename);    // 打开文件
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open file: " + filename);
            }
            std::string line;
            while (std::getline(file, line)) {  // 按行读取文件内容
                while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
                    line.pop_back();
                }
                lines.push_back(line);          // 将每行内容添加到 vector 中
            }
            file.close();  // 关闭文件
            return lines;
        }

        int read_data_from_file(const char *path, char **out_data)
        {
            FILE *fp = fopen(path, "rb");
            if(fp == nullptr) {
                STDEPLOY_ERROR("fopen %s fail!", path);
                return -1;
            }
            fseek(fp, 0, SEEK_END);
            int file_size = ftell(fp);
            char *data = (char *)malloc(file_size+1);
            data[file_size] = 0;
            fseek(fp, 0, SEEK_SET);
            if(file_size != fread(data, 1, file_size, fp)) {
                STDEPLOY_ERROR("fread %s fail!", path);
                free(data);
                fclose(fp);
                return -1;
            }
            if(fp) {
                fclose(fp);
            }
            *out_data = data;
            return file_size;
        }

        std::vector<float> loadFloatArrayFromBin(const std::string &filename, size_t size) {
            std::ifstream file(filename, std::ios::in | std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open file for reading: " << filename << std::endl;
                return {};
            }

            std::vector<float> data(size);
            file.read(reinterpret_cast<char *>(data.data()), size * sizeof(float));
            file.close();

            if (!file.good()) {
                std::cerr << "Failed to read data from file: " << filename << std::endl;
                data.clear(); // Optionally clear the vector if you want
            }
            return data;
        }

        void saveFloatArrayToBin(const float *data, size_t size, const std::string &filename) {
            std::ofstream file(filename, std::ios::out | std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open file for writing: " << filename << std::endl;
                return;
            }
            file.write(reinterpret_cast<const char *>(data), size * sizeof(float));
            file.close();
            std::cout << "Data written to " << filename << std::endl;
        }

        bool ReadBinaryFromFile(const std::string &file, std::string *contents) {
            std::ifstream fin(file, std::ios::in | std::ios::binary);
            if (!fin.is_open()) {
                stdeploy::SDERROR << "Failed to open file: " << file << " to read." << std::endl;
                return false;
            }
            fin.seekg(0, std::ios::end);
            contents->clear();
            contents->resize(fin.tellg());
            fin.seekg(0, std::ios::beg);
            fin.read(&(contents->at(0)), contents->size());
            fin.close();
            return true;
        }

        std::string path_join(const std::vector<std::string> &paths, const std::string &sep) {
            if (paths.size() == 1) {
                return paths[0];
            }
            std::string filepath = "";
            for (const auto &path : paths) {
                if (filepath == "") {
                    filepath += path;
                    continue;
                }
                if (path[0] == sep[0] || filepath.back() == sep[0]) {
                    filepath += path;
                } else {
                    filepath += sep + path;
                }
            }
            return filepath;
        }

        std::string path_join(const std::string &folder, const std::string &filename) {
            return path_join(std::vector<std::string>{folder, filename}, OS_PATH_SEP);
        }

        bool is_dir(const std::string &path) {
            struct stat buffer;
            return (stat(path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
        }

        std::string get_basename(const std::string &path) {
            // 查找最后一个路径分隔符的位置
            size_t idx = path.find_last_of(OS_PATH_SEP);
            // 如果没有找到分隔符，则返回整个路径
            if (idx == std::string::npos) {
                return path;
            }
            // 返回分隔符之后的部分
            return path.substr(idx + 1);
        }

        std::string get_basename_without_suffix(const std::string &path) {
            std::string basename = get_basename(path);
            // 查找最后一个点的位置，并确保它不是第一个字符（避免处理类似 ".hiddenfile" 的情况）
            size_t dot_idx = basename.find_last_of('.');
            if (dot_idx != std::string::npos && dot_idx != 0) {
                // 返回去掉后缀的文件名
                return basename.substr(0, dot_idx);
            }

            // 如果没有找到点或者点是第一个字符，则返回原始basename
            return basename;
        }

        std::string get_file_suffix(const std::string &path) {
            std::string name;
            int idx = path.find_last_of(".");
            if (idx > -1) {
                name = path.substr(idx + 1, path.length());
            }
            return name;
        }

        std::string get_parentDir_name(const std::string &path) {
            std::string dir_name;
            int idx = path.find_last_of(OS_PATH_SEP);
            if (idx > -1) {
                dir_name = path.substr(0, idx);
            }
            return dir_name;
        }

        std::string get_subDir_name(const std::string &path) {
            std::string parent_name = get_parentDir_name(path);
            std::string sub_name = get_basename(parent_name);
            return sub_name;
        }

        bool path_exists(const std::string &path) {
#ifdef _WIN32
            struct _stat buffer;
    return (_stat(path.c_str(),&buffer)==0);
#else
            struct stat buffer;
            return (stat(path.c_str(), &buffer) == 0);
#endif
        }

        void make_dir(const std::string &path) {
            if (path_exists(path))
                return;
            int ret = 0;
#ifdef _WIN32
            ret = _mkdir(path.c_str());
#else
            ret = mkdir(path.c_str(), 0755);
#endif
            if (ret != 0) {
                std::string path_error(path);
                path_error += "mkdir failed!";
                throw std::runtime_error(path_error);
            }
        }

        void make_dirs(const std::string &path) {
            if (path.empty())
                return;
            if (path_exists(path))
                return;
            make_dirs(get_parentDir_name(path));
            make_dir(path);
        }

        bool file_exist(const std::string &path) {
            auto flag = false;
            std::fstream fs(path, std::ios::in | std::ios::binary);
            flag = fs.is_open();
            fs.close();
            return flag;
        }

        bool read_file(const std::string &path, std::vector<char> &data) {
            std::fstream fs(path, std::ios::in | std::ios::binary);
            if (!fs.is_open()) {
                return false;
            }
            fs.seekg(std::ios::end);
            auto fs_end = fs.tellg();
            fs.seekg(std::ios::beg);
            auto fs_beg = fs.tellg();
            auto file_size = static_cast<size_t>(fs_end - fs_beg);
            auto vector_size = data.size();
            data.reserve(vector_size + file_size);
            data.insert(data.end(), std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>());
            fs.close();

            return true;
        }

        bool dump_file(const std::string &path, std::vector<char> &data) {
            std::fstream fs(path, std::ios::out | std::ios::binary);

            if (!fs.is_open() || fs.fail()) {
                STDEPLOY_ERROR("[ERR] cannot open file %s \n", path.c_str());
                return false;
            }
            fs.write(data.data(), data.size());
            return true;
        }

        bool dump_file(const std::string &path, char *data, int size) {
            std::fstream fs(path, std::ios::out | std::ios::binary);

            if (!fs.is_open() || fs.fail()) {
                STDEPLOY_ERROR("[ERR] cannot open file %s \n", path.c_str());
                return false;
            }

            fs.write(data, size);

            return true;
        }

        std::vector<std::string> split_string(const std::string &content, const std::string &delimiter) {
            std::vector<std::string> result;

            std::string::size_type pos1 = 0;
            std::string::size_type pos2 = content.find(delimiter);

            while (std::string::npos != pos2) {
                result.push_back(content.substr(pos1, pos2 - pos1));

                pos1 = pos2 + delimiter.size();
                pos2 = content.find(delimiter, pos1);
            }

            if (pos1 != content.length()) {
                result.push_back(content.substr(pos1));
            }

            return result;
        }

        std::string replace_file_ext(const std::string &file_path, std::string &new_ext_name) {
            size_t found_idx = file_path.rfind('.');
            if (found_idx != std::string::npos) {
                std::string new_file_path = file_path.substr(0, found_idx); // 获取文件名部分
                new_file_path += new_ext_name; // 添加新的扩展名
                return new_file_path;
            }
            return file_path;
        }


        bool is_supported_image_extension(const std::string &filename) {
            std::string file_ext = get_file_suffix(filename);
            if (!file_ext.empty()) {
                if (file_ext == "jpg" || file_ext == "jpeg" || file_ext == "bmp" || file_ext == "png") {
                    return true;
                } else {
                    return false;
                }
            } else {
                STDEPLOY_ERROR("%s file ext empty", file_ext.c_str());
                return false;
            }
        }

        bool load_img_files(const std::string &img_path, std::vector<std::string> &img_files) {
            if (is_dir(img_path)) {
                DIR *dir = opendir(img_path.c_str());
                if (dir == NULL) {
                    STDEPLOY_ERROR("opendir NULL");
                    return false;
                }
                struct dirent *entry;
                while ((entry = readdir(dir)) != NULL) {
                    if (entry->d_name[0] == '.')
                        continue;
                    std::string full_path = img_path + OS_PATH_SEP + entry->d_name;
                    // Check if it's a regular file using stat instead of d_type
                    struct stat path_stat;
                    if (stat(full_path.c_str(), &path_stat) != 0) {
                        SDERROR << "stat failed for path: " << full_path << std::endl;
                        continue;
                    }
                    if (!S_ISREG(path_stat.st_mode))
                        continue;
                    if (is_supported_image_extension(full_path)) {
                        img_files.emplace_back(full_path);
                    }
                }
                closedir(dir); // 关闭目录流
            } else {
                struct stat path_stat;
                if (stat(img_path.c_str(), &path_stat) == 0 && S_ISREG(path_stat.st_mode)) {
                    if (is_supported_image_extension(img_path)) {
                        img_files.emplace_back(img_path);
                    }
                } else {
                    STDEPLOY_ERROR("Path is not a directory or a IMG file: ");
                    return false;
                }
            }
            return true;
        }

        bool load_img_files_sort_by_number(const std::string &img_path, std::vector<std::string> &img_files) {
            if (is_dir(img_path)) {
                DIR *dir = opendir(img_path.c_str());
                if (dir == NULL) {
                    STDEPLOY_ERROR("opendir NULL");
                    return false;
                }
                struct dirent *entry;
                while ((entry = readdir(dir)) != NULL) {
                    if (entry->d_name[0] == '.')
                        continue;
                    std::string full_path = img_path + OS_PATH_SEP + entry->d_name;
                    // Check if it's a regular file using stat instead of d_type
                    struct stat path_stat;
                    if (stat(full_path.c_str(), &path_stat) != 0) {
                        SDERROR << "stat failed for path: " << full_path << std::endl;
                        continue;
                    }
                    if (!S_ISREG(path_stat.st_mode))
                        continue;
                    if (is_supported_image_extension(full_path)) {
                        img_files.emplace_back(full_path);
                    }
                }
                closedir(dir); // 关闭目录流
            } else {
                struct stat path_stat;
                if (stat(img_path.c_str(), &path_stat) == 0 && S_ISREG(path_stat.st_mode)) {
                    if (is_supported_image_extension(img_path)) {
                        img_files.emplace_back(img_path);
                    }
                } else {
                    STDEPLOY_ERROR("Path is not a directory or a IMG file: ");
                    return false;
                }
            }
            std::sort(img_files.begin(), img_files.end(), cmp_filenames_by_number);
            return true;
        }

        int read_yuv_data(const std::string &filename, unsigned char *data, int size) {
            std::ifstream file(filename, std::ios::in | std::ios::binary);
            if (!file) {
                return -1;
            }
            file.read(reinterpret_cast<char *>(data), size);
            file.close();
            return 0;
        }

        int save_yuv_data(const std::string &filename, const unsigned char *data, int size) {
            // 打开文件以二进制写入模式
            std::ofstream file(filename, std::ios::out | std::ios::binary);
            if (!file) {
                // 如果无法打开文件，则返回错误码
                return -1;
            }

            // 写入数据到文件
            file.write(reinterpret_cast<const char *>(data), size);

            // 检查写入是否成功
            if (!file.good()) {
                file.close();
                return -1;
            }

            // 关闭文件并返回成功码
            file.close();
            return 0;
        }

        bool cmp_filenames_by_number(const std::string &a, const std::string &b) {
            int numA = 0, numB = 0;
            // 解析文件名，提取数字和后缀
            std::string basename_a = get_basename(a);
            std::string basename_b = get_basename(b);
            numA = std::stoi(basename_a.substr(0, basename_a.rfind(".")));
            numB = std::stoi(basename_b.substr(0, basename_b.rfind(".")));
            return numA < numB;
        }

    } //namespace utils
} //namespace stdeploy

