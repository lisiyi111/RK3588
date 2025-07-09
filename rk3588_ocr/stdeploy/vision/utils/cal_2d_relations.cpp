/**
  *************************************************
  * @file               :cal_2d_relations.cpp
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2025/2/21                
  *************************************************
  */

#include "stdeploy/vision/utils/utils.h"

namespace stdeploy {
    namespace vision {
        namespace utils {

            std::vector<cv::Point2f> sort_points_clockwise(const std::vector<cv::Point2f> &points) {
                if (points.size() != 4) {
                    STDEPLOY_ERROR("Sort_points_clockwise failed, expected exactly 4 points");
                    return points;
                }
                std::vector<cv::Point2f> sortedPoints = points;
                std::vector<cv::Point2f> ordered_pts(4);
                // 对所有点按y坐标升序，x坐标升序排序
                std::sort(sortedPoints.begin(), sortedPoints.end(), comparePoints);
                // sortedPoints[0] - [1] 是上面的点
                // x小的为左上角的点，x大的右上角的点
                if (sortedPoints[0].x < sortedPoints[1].x) {
                    ordered_pts[0] = sortedPoints[0]; // 左上角
                    ordered_pts[1] = sortedPoints[1]; // 右上角
                } else {
                    ordered_pts[0] = sortedPoints[1]; // 左上角
                    ordered_pts[1] = sortedPoints[0]; // 右上角
                }
                // 剩下的两个点中，x坐标较大的为右下角，较小的为左下角
                if (sortedPoints[2].x > sortedPoints[3].x) {
                    ordered_pts[2] = sortedPoints[2]; // 右下角
                    ordered_pts[3] = sortedPoints[3]; // 左下角
                } else {
                    ordered_pts[2] = sortedPoints[3]; // 右下角
                    ordered_pts[3] = sortedPoints[2]; // 左下角
                }
                return ordered_pts;
            }


            bool comparePoints(const cv::Point2f &a, const cv::Point2f &b) {
                // 首先按y坐标排序，如果y坐标相同，则按x坐标排序
                if (a.y == b.y)
                    return a.x < b.x;
                else
                    return a.y < b.y;
            }


            bool cmp_area_contour(const std::vector<cv::Point> &c1, const std::vector<cv::Point> &c2) {
                return cv::contourArea(c1) > cv::contourArea(c2);
            }


        }//namespace utils
    }//namespace vision
}//namespace stdeploy