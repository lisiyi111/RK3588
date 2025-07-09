/**
  *************************************************
  * @file               :test_runtime.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/6                
  *************************************************
  */

#include "stdeploy/vision.h"

namespace sd = stdeploy;


void show_help(char *msg) {
    printf("------------------------------------------------ Help ----------------------------------------\n");
    printf("Usage : %s args... \n", msg);
    printf("Please Input:\n");
    printf("\t 1) det model file path: e.g  tests.xxx \n");
    printf("\t 2) det cfg file path: e.g  tests.xxx \n");
    printf("\t 3) cls model file path: e.g  tests.xxx \n");
    printf("\t 4) cls cfg file path: e.g  tests.xxx \n");
    printf("\t 5) rec model file path: e.g  tests.xxx \n");
    printf("\t 6) rec cfg file path: e.g  tests.xxx \n");
    printf("\t 7) tests img path: e.g ../imgs \n");
    printf("\t 8) number: e.g warm up number \n");
    printf("---------------------------------------------------------------------------------------------\n");
}


int main(int argc, char **argv) {

    if (argc != 9) {
        show_help(argv[0]);
        return -1;
    }

    std::string det_model_file = std::string(argv[1]);
    std::string det_config_file = std::string(argv[2]);
    std::string cls_model_file = std::string(argv[3]);
    std::string cls_config_file = std::string(argv[4]);
    std::string rec_model_file = std::string(argv[5]);
    std::string rec_config_file = std::string(argv[6]);
    std::string image_path = std::string(argv[7]);
    int warm_up_number = atoi(argv[8]);

    sd::SetLogger(true, true);

    sd::RuntimeOption runtime_option;
    runtime_option.UseRKNPU2();

    auto *det_model = new sd::vision::ocr::DBNet(det_model_file, "", runtime_option,
                                                 sd::ModelFormat::rknn, det_config_file);
    auto *cls_model = new sd::vision::ocr::TextAngleCls(cls_model_file, "", runtime_option,
                                                        sd::ModelFormat::rknn, cls_config_file);
    auto *rec_model = new sd::vision::ocr::CRNN(rec_model_file, "", runtime_option,
                                                sd::ModelFormat::rknn, rec_config_file);

    auto *ocr_model = new sd::vision::pipeline::PPOCRv4(det_model, cls_model, rec_model);
    if (!ocr_model->Initialized()) {
        std::cerr << "Failed to initialize" << std::endl;
        delete ocr_model;
        std::exit(-1);
    }
    std::vector<std::string> img_files;
    sd::utils::load_img_files(image_path, img_files);
    std::string output_img_path = "output_img/";
    sd::utils::make_dir(output_img_path);

    for (int w = 0; w < warm_up_number; w++) {
        for (size_t i = 0; i < img_files.size(); i++) {
            STDEPLOY_INFO("Infer file: %s", img_files[i].c_str());
            std::string img_basename = sd::utils::get_basename(img_files[i]);
            std::string dst_img_path = sd::utils::path_join(output_img_path, img_basename);
            cv::Mat img = cv::imread(img_files[i]);
            if (img.empty()) {
                STDEPLOY_ERROR("Read img failed.");
                return -1;
            }
            sd::vision::OCRResult res;
            ocr_model->Predict(img, &res);
            std::cout << res.Str() << std::endl;
            cv::Mat show_img = stdeploy::vision::VisOcr(img, res);
            cv::imwrite(dst_img_path, show_img);
        }
    }

    delete ocr_model;
    return 0;
}
