//
// Created by zhuzf on 2023/4/28.
//

#include "stdeploy/vision/visualize/visualize.h"
#include "stdeploy/utils/log_util.h"

namespace stdeploy {
    namespace vision {




        cv::Mat VisDepth(const cv::Mat &im,
                         const DepthResult &result) {
            cv::Mat depth_map(result.shape[0], result.shape[1], CV_8UC1,
                              const_cast<uint8_t *>(result.vis_depth.data()));
            cv::Mat depth_colored;
            cv::applyColorMap(depth_map, depth_colored, cv::COLORMAP_JET);
            cv::Mat depth_resized;
//            cv::resize(depth_colored, depth_resized, cv::Size(im.cols, im.rows), cv::INTER_CUBIC);
            // INTER_NEAREST不修改像素值，防止缩放导致深度值异常
            cv::resize(depth_colored, depth_resized, cv::Size(im.cols, im.rows), cv::INTER_NEAREST);
            return depth_resized;
        }


        cv::Mat
        VisKeypointDetection(const cv::Mat &im, const KeyPointDetectionResult &results,
                             float conf_threshold) {
            const int edge[][2] = {{0,  1},
                                   {0,  2},
                                   {1,  3},
                                   {2,  4},
                                   {3,  5},
                                   {4,  6},
                                   {5,  7},
                                   {6,  8},
                                   {7,  9},
                                   {8,  10},
                                   {5,  11},
                                   {6,  12},
                                   {11, 13},
                                   {12, 14},
                                   {13, 15},
                                   {14, 16},
                                   {11, 12}};
            auto colormap = GenerateColorMap(1000);
            cv::Mat vis_img = im.clone();
            int detection_nums = results.keypoints.size() / 17;
            for (int i = 0; i < detection_nums; i++) {
                int index = i * 17;
                bool is_over_threshold = true;
                for (int j = 0; j < results.num_joints; j++) {
                    if (results.scores[index + j] < conf_threshold) {
                        is_over_threshold = false;
                        break;
                    }
                }
                if (is_over_threshold) {
                    for (int k = 0; k < results.num_joints; k++) {
                        int x_coord = int(results.keypoints[index + k][0]);
                        int y_coord = int(results.keypoints[index + k][1]);
                        cv::circle(vis_img, cv::Point2d(x_coord, y_coord), 1,
                                   cv::Scalar(0, 0, 255), 2);
                        int x_start = int(results.keypoints[index + edge[k][0]][0]);
                        int y_start = int(results.keypoints[index + edge[k][0]][1]);
                        int x_end = int(results.keypoints[index + edge[k][1]][0]);
                        int y_end = int(results.keypoints[index + edge[k][1]][1]);
                        cv::line(vis_img, cv::Point2d(x_start, y_start),
                                 cv::Point2d(x_end, y_end), colormap[k], 1);
                    }
                }
            }
            return vis_img;
        }


        cv::Mat VisOcr(const cv::Mat &im, const OCRResult &ocr_result) {
            auto vis_im = im.clone();

            for (int n = 0; n < ocr_result.boxes.size(); n++) {
                cv::Point rook_points[4];

                for (int m = 0; m < 4; m++) {
                    rook_points[m] = cv::Point(int(ocr_result.boxes[n][m * 2]),
                                               int(ocr_result.boxes[n][m * 2 + 1]));
                }
                const cv::Point *ppt[1] = {rook_points};
                int npt[] = {4};
                cv::polylines(vis_im, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
            }
            return vis_im;
        }


    } //namespace vision
} //namespace stdeploy