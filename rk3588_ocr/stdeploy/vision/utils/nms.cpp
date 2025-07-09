/**
  *************************************************
  * @file               :nms.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/26                
  *************************************************
  */


#include "stdeploy/vision/utils/utils.h"


namespace stdeploy {
    namespace vision {
        namespace utils {

            float cal_iou(const Object &a, const Object &b) {
                cv::Rect r1 = cv::Rect(cv::Point(a.xmin, a.ymin), cv::Point(a.xmax, a.ymax));
                cv::Rect r2 = cv::Rect(cv::Point(b.xmin, b.ymin), cv::Point(b.xmax, b.ymax));
                cv::Rect inter = r1 & r2;
                if (inter.area() <= 0)
                    return 0.;
                float iou_value = 1. * inter.area() / (r1.area() + r2.area() - inter.area());
                return iou_value;
            }


            int nms_sort_boxes(std::vector<Object> &objects, std::vector<int> &keptIndices, float nms_thresh) {
                keptIndices.clear();
                std::sort(objects.begin(), objects.end(), cmp_score);
                for (int i = 0; i < objects.size(); i++) {
                    int keep = 1;
                    for (int j = 0; j < keptIndices.size(); j++) {
                        float iou_value = cal_iou(objects[i], objects[keptIndices[j]]);
                        if (iou_value >= nms_thresh) {
                            keep = 0;
                        }
                    }
                    if (keep) {
                        keptIndices.push_back(i);
                    }
                }
                return 0;
            }


            void GetCovarianceMatrix(const ObbBox &Box, float &A, float &B, float &C) {
                float a = Box.width;
                float b = Box.height;
                float c = Box.angle;

                float cos1 = cos(c);
                float sin1 = sin(c);
                float cos2 = pow(cos1, 2);
                float sin2 = pow(sin1, 2);

                A = a * cos2 + b * sin2;
                B = a * sin2 + b * cos2;
                C = (a - b) * cos1 * sin1;
            }

            float cal_probiou(const ObbBox &a, const ObbBox &b) {
                float eps = 1e-7;

                float x1 = a.cx;
                float y1 = a.cy;
                float x2 = b.cx;
                float y2 = b.cy;

                float a1 = 0, b1 = 0, c1 = 0;
                GetCovarianceMatrix(a, a1, b1, c1);

                float a2 = 0, b2 = 0, c2 = 0;
                GetCovarianceMatrix(b, a2, b2, c2);

                float t1 = (((a1 + a2) * pow((y1 - y2), 2) + (b1 + b2) * pow((x1 - x2), 2)) /
                            ((a1 + a2) * (b1 + b2) - pow((c1 + c2), 2) + eps)) * 0.25;
                float t2 =
                        (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - pow((c1 + c2), 2) + eps)) * 0.5;

                float temp1 = (a1 * b1 - pow(c1, 2));
                temp1 = temp1 > 0 ? temp1 : 0;

                float temp2 = (a2 * b2 - pow(c2, 2));
                temp2 = temp2 > 0 ? temp2 : 0;

                float t3 =
                        log((((a1 + a2) * (b1 + b2) - pow((c1 + c2), 2)) / (4 * sqrt((temp1 * temp2)) + eps) + eps)) *
                        0.5;
                float bd = 0;
                if ((t1 + t2 + t3) > 100) {
                    bd = 100;
                } else if ((t1 + t2 + t3) < eps) {
                    bd = eps;
                } else {
                    bd = t1 + t2 + t3;
                }

                float hd = sqrt((1.0 - exp(-bd) + eps));
                return 1 - hd;
            }

            int rotate_nms_sort_boxes(std::vector<ObbBox> &objects, std::vector<int> &keptIndices, float nms_thresh) {
                keptIndices.clear();
                std::sort(objects.begin(), objects.end(), cmp_score);
                for (int i = 0; i < objects.size(); i++) {
                    int keep = 1;
                    for (int j = 0; j < keptIndices.size(); j++) {
                        float iou_value = cal_probiou(objects[i], objects[keptIndices[j]]);
                        if (iou_value >= nms_thresh) {
                            keep = 0;
                        }
                    }
                    if (keep) {
                        keptIndices.push_back(i);
                    }
                }
                return 0;
            }

            std::array<float, 8> rbbox_to_corners(const std::vector<float> &rbbox) {
                // list {cx, cy, w, h, angle1}

                std::array<float, 8> corners{};
                float x = rbbox[0];
                float y = rbbox[1];
                float w = rbbox[2];
                float h = rbbox[3];
                float angle = rbbox[4];
                // 是否需要根据数据标注的方法，调整角点的顺序
//                float bw_ = cw > ch ? cw : ch;
//                float bh_ = cw > ch ? ch : cw;
//                float bt = cw > ch ? (angle - float(int(angle / pi)) * pi) : ((angle + pi / 2) - float(int((angle + pi / 2) / pi)) * pi);
//
                float cos_value = cos(angle);
                float sin_value = sin(angle);

                float vec1x = w / 2 * cos_value;
                float vec1y = w / 2 * sin_value;
                float vec2x = -h / 2 * sin_value;
                float vec2y = h / 2 * cos_value;

                corners[0] = x + vec1x + vec2x;
                corners[1]  = y + vec1y + vec2y;

                corners[2]  = x + vec1x - vec2x;
                corners[3]  = y + vec1y - vec2y;

                corners[4]  = x - vec1x - vec2x;
                corners[5]  = y - vec1y - vec2y;

                corners[6]  = x - vec1x + vec2x;
                corners[7]  = y - vec1y + vec2y;
                return corners;
            }


        }//namespace utils
    }//namespace vision
}//namespace stdeploy

