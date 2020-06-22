/*  Copyright (C) <2020>  <Yong WU>
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.
    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#ifdef CAFFE_CLASSIFIER_LINK_SHARED
#if defined(__GUNC__) && __GNUC__ >= 4
#define CAFFE_CLASSIFIER_API __attribute__ ((bisubility("default")))
#elif defined(__GUNC__)
#define CAFFE_CLASSIFIER_API
#elif defined(_MSC_VER)
#if defined (CAFFE_CLASSIFIER_API_EXPORTS)
#define CAFFE_CLASSIFIER_API __declspec(dllexport)
#else
#define CAFFE_CLASSIFIER_API __delcspec(dllimport)
#endif
#else
#define CAFFE_CLASSIFIER_API
#endif
#else
#define CAFFE_CLASSIFIER_API
#endif

#include <vector>
#include <string>

#include "opencv2/core/core.hpp"
#include "bbox.hpp"

struct Params {
  float resize_factor = 0.709f;
  float confidence_thresh[3] = { 0.6f, 0.7f, 0.7f };  // pnet, rnet, onet
  float nms_thresh1 = 0.5f;                         // intra-scale num threshold
  float nms_thresh = 0.7f;
  float mean_value = 127.5f;
  float normalize_factor = 0.0078125f;
  int min_face_size = 20;
  std::vector<std::string> offset_name = { "conv4-2", "conv5-2", "conv6-2" };
  std::string prob_name = "prob1";
  std::string landmark_name = "conv6-3";
  std::string input_name = "data";
  const int stage_num = 3;
  const int cell_size = 12;
  const int stride_size = 2;
  const int landmark_num = 5;
};

class CAFFE_CLASSIFIER_API CaffeMTCNN {
public:
  static CaffeMTCNN* NewMTCNN(const std::vector<std::string>& model_files,
                         const std::vector<std::string>& trained_files, const Params& params);
  CaffeMTCNN(const Params& params): params_(params) {}
  CaffeMTCNN(const CaffeMTCNN&) = delete;
  CaffeMTCNN& operator=(const CaffeMTCNN&) = delete;
  virtual ~CaffeMTCNN() = default;
  
  virtual std::vector<BBox> Detect(const cv::Mat& img) = 0;
  Params& params() { return params_; }

protected:
  Params params_;
};