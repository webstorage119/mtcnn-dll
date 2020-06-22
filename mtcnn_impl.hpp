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
#include "caffe_mtcnn.hpp"
#include "caffe/caffe.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <memory>
#include <vector>
#include <string>

class MTCNNImpl : public CaffeMTCNN {
public:
  MTCNNImpl(const std::vector<std::string>& model_files,
            const std::vector<std::string>& trained_files, const Params& params);

  std::vector<BBox> Detect(const cv::Mat& img) override;

private:
  void WrapInputLayer(int stage, int n, std::vector<cv::Mat>* input_channels);
  std::vector<float> PyramidImage(const cv::Size& img_size);
  void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
  void NetInference(const std::vector<cv::Mat>& imgs, const int stage);
  void PNetProcess(const cv::Mat& img, std::vector<BBox>& bboxes);
  void RNetProcess(const cv::Mat& img, std::vector<BBox>& bboxes);
  void ONetProcess(const cv::Mat& img, std::vector<BBox>& bboxes);

  std::vector<std::shared_ptr<caffe::Net<float>>> nets_;
  std::vector<cv::Size2i> input_geometry_;
  std::vector<int> num_channels_;
};