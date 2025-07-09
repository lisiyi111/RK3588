/**
  *************************************************
  * @file               :vision.h
  * @author             :zzf
  * @brief              :None
  * @attention          :None
  * @date               :2023/4/6                
  *************************************************
  */

#pragma once

#ifdef ENABLE_VISION

#include "stdeploy/vision/visualize/visualize.h"
#include "stdeploy/vision/common/vision_struct.h"
#include "stdeploy/utils/file_util.h"

#ifdef ENABLE_VISION_CLASSIFICATION
#include "stdeploy/vision/classification/model.h"
#endif

#ifdef ENABLE_VISION_DETECTION
#include "stdeploy/vision/detection/model.h"
#endif

#ifdef ENABLE_VISION_OCR
#include "stdeploy/vision/ocr/dbnet.h"
#include "stdeploy/vision/ocr/crnn.h"
#include "stdeploy/vision/ocr/ppocr/text_angle_cls.h"
#include "stdeploy/vision/ocr/ppocr/ppocrv2.h"
#include "stdeploy/vision/ocr/ppocr/ppocrv3.h"
#include "stdeploy/vision/ocr/ppocr/ppocrv4.h"
#endif

#ifdef ENABLE_VISION_SEGMENTATION
#include "stdeploy/vision/segmentation/model.h"
#endif

#ifdef ENABLE_VISION_KEYPOINT
#include "stdeploy/vision/keypoint/model.h"
#endif

#ifdef ENABLE_VISION_TRACKING
#include "stdeploy/vision/tracking/bytetrack/BYTETracker.h"
#endif

#ifdef ENABLE_VISION_FACE
#include "stdeploy/vision/face/face_det.h"
#include "stdeploy/vision/face/face_id.h"
#include "stdeploy/vision/face/face_recognition.h"
#endif

#ifdef ENABLE_VISION_CONTRIB
#include "stdeploy/vision/contrib/vision_contrib.h"
#endif

#ifdef ENABLE_VISION_DEPTH
#include "stdeploy/vision/depth/model.h"
#endif

#ifdef ENABLE_VISION_SR
#include "stdeploy/vision/sr/real_esrgan.h"
#endif

#endif



