/**
  *************************************************
  * @file               :model.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/14                
  *************************************************
  */
#pragma once

#include "stdeploy/stdeploy_model.h"
#include "stdeploy/vision/classification/postprocessor.h"
#include "stdeploy/vision/common/image_processes/preprocessor.h"

namespace stdeploy {
    namespace vision {
        namespace classification {

            class STDEPLOY_DECL ClsModel : public StDeployModel {
            public:
                ClsModel(const std::string &model_file,
                         const std::string &params_file = "",
                         const RuntimeOption &custom_option = RuntimeOption(),
                         const ModelFormat &model_format = ModelFormat::onnx,
                         const std::string &config_file = "");

                ~ClsModel() override;

                std::string ModelName() const override { return "Cls"; }

                /** \brief Predict cv::Mat
                *
                * \param[in] img: cv::Mat data
                * \param[in] result: ClassifyResult format output
                * \return bool: if the postprocess true, otherwise false
                */
                bool Predict(sd::Mat &img, ClassifyResult *result);

                /** \brief preprocessor to get function
                 *
                 * \return preprocessor_
                 */
                BasePreprocessor &GetPreprocessor() {
                    return *preprocessor_;
                }

                /** \brief postprocessor_ to get function
                 *
                 * \return postprocessor_
                 */
                ClsPostprocessor &GetPostprocessor() {
                    return *postprocessor_;
                }

            protected:
                bool Init();

                BasePreprocessor *preprocessor_ = nullptr;
                ClsPostprocessor *postprocessor_ = nullptr;

            private:
                std::string config_file_;
            };


        } //namespace classification
    } //namespace vision
} //namespace stdeploy