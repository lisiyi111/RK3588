/**
  *************************************************
  * @file               :detection_struct.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/8/24                
  *************************************************
  */

#pragma once

#include <vector>

namespace stdeploy {
    namespace vision {

        // struct for rect box
        typedef struct Object {
            float xmin{};
            float ymin{};
            float xmax{};
            float ymax{};
            int label_id{};
            float score{};
            std::vector<float> masks;
            std::vector<float> key_points;
        } Object;

        // struct for rotate box
        typedef struct ObbBox : Object {
            explicit ObbBox(float cx = 0,
                            float cy = 0,
                            float w = 0,
                            float h = 0,
                            float score = 0,
                            int label_id = -1,
                            float angle = 0) {
                this->xmin = cx - w / 2;
                this->ymin = cy - h / 2;
                this->xmax = cx + w / 2;
                this->ymax = cy + h / 2;
                this->score = score;
                this->label_id = label_id;
                this->angle = angle;
                this->cx = cx;
                this->cy = cy;
                this->width = w;
                this->height = h;
            }

            float angle;
            float cx;
            float cy;
            float width;
            float height;
        } ObbBox;

        // struct for track box
        typedef struct TrackBox : Object {
            explicit TrackBox(float x = 0,
                              float y = 0,
                              float w = 0,
                              float h = 0,
                              float score = 0,
                              int class_id = -1,
                              int track_id = -1) {
                this->xmin = x;
                this->ymin = y;
                this->xmax = x + w;
                this->ymax = y + h;
                this->width = w;
                this->height = h;
                this->score = score;
                this->label_id = class_id;
                this->track_id = track_id;
            }

            int track_id;
            float width;
            float height;
        } TrackBox;

        // struct for rect box
        typedef struct PreprocessParams {
            // base
            int src_batch = 1;                  // src img batch
            int src_width = 640;                  // src img width
            int src_height = 640;                 // src img height
            int src_channel = 3;                // src img channel
            int dst_batch = 1;                  // dst img batch
            int dst_width = 640;                  // dst img width
            int dst_height = 640;                 // dst img height
            int dst_channel = 3;                // dst img channel
            // letterbox resize
            bool is_letterbox = false;
            float scale_width = 1.0;           // resize scale width
            float scale_height = 1.0;             // resize scale height
            int pad_width_left = 0;             // pad left
            int pad_width_right = 0;            // pad right
            int pad_height_top = 0;             // pad top
            int pad_height_bottom = 0;          // pad bottom
            int resize_width = 0;               // letterbox resize width
            int resize_height = 0;              // letterbox resize height
            // crop and resize
            bool is_crop_resize = false;
            int src_box_x = 0;                  // crop src x
            int src_box_y = 0;                  // crop src y
            int src_box_w = 0;                  // crop src w
            int src_box_h = 0;                  // crop src h
            int dst_box_x = 0;                  // crop dst x
            int dst_box_y = 0;                  // crop dst y
            int dst_box_w = 0;                  // crop dst w
            int dst_box_h = 0;                  // crop dst h
        } PreprocessParams;


    } //namespace vision
} //namespace stdeploy


