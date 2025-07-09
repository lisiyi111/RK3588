/**
  *************************************************
  * @file               :result.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/14                
  *************************************************
  */

#pragma once

#include "stdeploy/stdeploy_model.h"
#include <opencv2/core/core.hpp>
#include <set>

namespace stdeploy {
    namespace vision {

        // result type
        enum STDEPLOY_DECL ResultType {
            UNKNOWN_RESULT,     // < un-known
            CLASSIFY,           // < 图像分类结果
            DETECTION,          // < 目标检测结果
            MASK,               // < 掩膜结果
            KEYPOINT_DETECTION, // < 关键点检测结果
            OCR,                // < OCR结果
            SEGMENTATION,       // < 语义分割结果
            FACE_RECOGNITION,   // < 人脸识别结果
            FEATURE_MAP,        // < 特征提取结果
            DEPTH,              // < 单目深度图结果
        };

        /**
         * @brief base result type
         */
        struct STDEPLOY_DECL BaseResult {
            ResultType type = ResultType::UNKNOWN_RESULT;
        };

        /**
         * @brief image classify result
         */
        struct STDEPLOY_DECL ClassifyResult : public BaseResult {

            ClassifyResult() = default;                                             // < 默认构造函数

            ClassifyResult(const ClassifyResult &other) = default;                  // < 默认拷贝构造函数
            ClassifyResult &operator=(const ClassifyResult &other) = default;       // < 默认拷贝赋值运算符

            ClassifyResult(ClassifyResult &&other) noexcept = default;              // < 默认移动构造函数
            ClassifyResult &operator=(ClassifyResult &&other) noexcept = default;   // < 默认移动赋值运算符
            // list: label ids
            std::vector<int32_t> label_ids;
            // list: label score
            std::vector<float> scores;
            // result type
            ResultType type = ResultType::CLASSIFY;
            // resize data
            void Resize(int size);
            // clear data
            void Clear();
            // free data
            void Free();
            // print data
            std::string Str();
        };


        /**
         * @brief image mask result
         */
        struct STDEPLOY_DECL Mask : public BaseResult {
            // Mask data buffer
            std::vector<uint8_t> data;
            // Shape of mask
            std::vector<int64_t> shape;

            ResultType type = ResultType::MASK;

            /// clear Mask result
            void Clear();

            /// Clear Mask result and free the memory
            void Free();

            /// Return a mutable pointer of the mask data buffer
            void *Data() { return data.data(); }

            /// Return a pointer of the mask data buffer for read only
            const void *Data() const { return data.data(); }

            /// Reserve size for mask data buffer
            void Reserve(int size);

            /// Resize the mask data buffer
            void Resize(int size);

            /// Debug function, convert the result to string to print
            std::string Str();
        };


        /*! @brief KeyPoint Detection result structure for all the keypoint detection models
         */
        struct STDEPLOY_DECL KeyPointDetectionResult : public BaseResult {
            /** \brief All the coordinates of detected keypoints for an input image, the size of `keypoints` is num_detected_objects * num_joints, and the element of `keypoint` is a array of 2 float values, means [x, y]
             */
            std::vector<std::array<float, 2>> keypoints;
            //// The confidence for all the detected points
            std::vector<float> scores;
            //// Number of joints for a detected object
            int num_joints = -1;

            ResultType type = ResultType::KEYPOINT_DETECTION;

            /// Clear KeyPointDetectionResult
            void Clear();

            /// Clear KeyPointDetectionResult and free the memory
            void Free();

            void Reserve(int size);

            void Resize(int size);

            /// Debug function, convert the result to string to print
            std::string Str();
        };


        /*! @brief Detection result structure for all the object detection models and instance segmentation models
        */
        struct STDEPLOY_DECL DetectionResult : public BaseResult {
            DetectionResult() = default;

            /** \brief All the detected object boxes for an input image, the size of `boxes` is the number of detected objects, and the element of `boxes` is a array of 4 float values, means [xmin, ymin, xmax, ymax]
             */
            std::vector<std::array<float, 4>> boxes;
            /** \brief The confidence for all the detected objects
             */
            std::vector<float> scores;
            /// The classify label for all the detected objects
            std::vector<int32_t> label_ids;
            /** \brief For instance segmentation model, `masks` is the predict mask for all the deteced objects
             */
            std::vector<Mask> masks;

            bool contain_rbox = false;
            std::vector<std::array<float, 8>> rboxes;
            std::vector<float> angles;

            /** \brief For instance segmentation model, `masks` is the predict mask for all the deteced objects
             * */
            /// Shows if the DetectionResult has mask
            bool contain_kpts = false;
            std::vector<KeyPointDetectionResult> kpts;

            /// Shows if the DetectionResult has mask
            bool contain_masks = false;

            ResultType type = ResultType::DETECTION;


            /// Clear DetectionResult
            void Clear();

            /// Clear DetectionResult and free the memory
            void Free();

            void Reserve(int size);

            void Resize(int size);

            /// Debug function, convert the result to string to print
            std::string Str();
        };

        struct STDEPLOY_DECL OCRResult : public BaseResult {
            std::vector<std::array<int, 8>> boxes;

            std::vector<std::string> text;
            std::vector<float> rec_scores;

            std::vector<float> cls_scores;
            std::vector<int32_t> cls_labels;

            ResultType type = ResultType::OCR;

            void Clear();

            std::string Str();
        };


        struct STDEPLOY_DECL SegmentationResult : public BaseResult {
            SegmentationResult() = default;

            /** \brief
             * `label_map` stores the pixel-level category labels for input image. the number of pixels is equal to label_map.size()
            */
            std::vector<uint8_t> label_map;
            /** \brief
             * `score_map` stores the probability of the predicted label for each pixel of input image.
            */
            std::vector<float> score_map;
            /// The output shape, means [H, W]
            std::vector<int64_t> shape;
            /// SegmentationResult whether containing score_map
            bool contain_score_map = false;

            /// Copy constructor
            SegmentationResult(const SegmentationResult &other) = default;

            /// Move assignment
            SegmentationResult &operator=(SegmentationResult &&other);

            ResultType type = ResultType::SEGMENTATION;

            /// Clear Segmentation result
            void Clear();

            /// Clear Segmentation result and free the memory
            void Free();

            void Reserve(int size);

            void Resize(int size);

            /// Debug function, convert the result to string to print
            std::string Str();
        };


        /*! @brief Feature map result structure for all the extractor models
        */
        struct STDEPLOY_DECL FeatureMapResult : public BaseResult {

            std::vector<float> embedding;

            ResultType type = ResultType::FEATURE_MAP;

            FeatureMapResult() = default;

            FeatureMapResult(const FeatureMapResult &res);

            /// Clear FaceRecognitionResult
            void Clear();

            /// Clear FaceRecognitionResult and free the memory
            void Free();

            void Reserve(int size);

            void Resize(int size);

            /// Debug function, convert the result to string to print
            std::string Str();

        };


        /*! @brief Depth map result structure for all the Depth models
*/
        struct STDEPLOY_DECL DepthResult : public BaseResult {

            std::vector<uint8_t> vis_depth;

            std::vector<float> relative_depth;

            /// The output shape, means [H, W]
            std::vector<int64_t> shape;

            float min_val;

            float max_val;

            ResultType type = ResultType::DEPTH;

            DepthResult() = default;

            void Clear();

            void Free();

            void Reserve(int size);

            void Resize(int size);

            /// Debug function, convert the result to string to print
            std::string Str();

        };


        struct STDEPLOY_DECL FaceRecognitionResult : public BaseResult {
            FaceRecognitionResult() = default;

            ResultType type = ResultType::FACE_RECOGNITION;
            // face detect
            std::vector<std::array<float, 4>> boxes{};
            std::vector<float> scores{};

            // face landmark
            /// Shows if the DetectionResult has mask
            bool contain_landmarks = false;
            std::vector<KeyPointDetectionResult> landmarks{};

            // face id
            /// Shows if the DetectionResult has embedding
            bool contain_embeddings = false;
            std::vector<FeatureMapResult> embeddings{};
            std::vector<std::string> names{};

            /// Clear face result
            void Clear();

            /// Clear face result and free the memory
            void Free();

            void Reserve(int size);

            void Resize(int size);

            /// Debug function, convert the result to string to print
            std::string Str();
        };


    } // namespace vision
} // namespace stdeploy