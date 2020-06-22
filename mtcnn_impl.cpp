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

#include "caffe_mtcnn.hpp"
#include "mtcnn_impl.hpp"
#include "utils.hpp"
#include "caffe/caffe.hpp"
#include <algorithm>

CaffeMTCNN* CaffeMTCNN::NewMTCNN(const std::vector<std::string>& model_files,
                     const std::vector<std::string>& trained_files, const Params& params) {
  return new MTCNNImpl(model_files, trained_files, params);
}

MTCNNImpl::MTCNNImpl(const std::vector<std::string>& model_files,
             const std::vector<std::string>& trained_files,
             const Params& params)
  : CaffeMTCNN(params) {
  CHECK_EQ(model_files.size(), trained_files.size());
  CHECK_EQ(model_files.size(), params_.stage_num);
#ifdef CPU_ONLY
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
  nets_.resize(params_.stage_num);
  for (int i = 0; i < params_.stage_num; ++i) {
    nets_[i].reset(new caffe::Net<float>(model_files[i], caffe::TEST));
    nets_[i]->CopyTrainedLayersFrom(trained_files[i]);
    caffe::Blob<float>* input_layer =
      nets_[i]->blob_by_name(params_.input_name).get();
    input_geometry_.emplace_back(input_layer->width(), input_layer->height());
    num_channels_.push_back(input_layer->channels());
    CHECK_EQ(num_channels_[i], 3)
      << "Input layer should have 3 channels.";
  }
}

std::vector<BBox> MTCNNImpl::Detect(const cv::Mat& img) {
  cv::Mat sample;
  cv::cvtColor(img, sample, cv::COLOR_BGR2RGB);
  sample.convertTo(sample, CV_32FC3);
  sample = sample.t();
  std::vector<BBox> bboxes;
  PNetProcess(sample, bboxes);
  if (bboxes.empty()) {
    return bboxes;
  }
  RNetProcess(sample, bboxes);
  if (bboxes.empty()) {
    return bboxes;
  }
  ONetProcess(sample, bboxes);
  // MATLAB to OpenCV Cord
  for (auto& bbox : bboxes) {
    std::swap(bbox.tl.x, bbox.tl.y);
    std::swap(bbox.br.x, bbox.br.y);
    for (auto& landmark : bbox.landmarks) {
      std::swap(landmark.x, landmark.y);
    }
  }
  return bboxes;
}

void MTCNNImpl::WrapInputLayer(int stage, int n,
                           std::vector<cv::Mat>* input_channels) {
  caffe::Blob<float>* input_layer = nets_[stage - 1]->blob_by_name(params_.input_name).get();;
  int width = input_layer->width();
  int height = input_layer->height();
  int channels = input_layer->channels();
  float* input_data =
    input_layer->mutable_cpu_data() + n * height * width * channels;
  for (int i = 0; i < channels; ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

std::vector<float> MTCNNImpl::PyramidImage(const cv::Size& img_size) {
  std::vector<float> scales;
  float scale = float(params_.cell_size) / params_.min_face_size;
  float min_scale = std::min(img_size.height, img_size.width) * scale;
  while (min_scale >= params_.cell_size) {
    scales.push_back(scale);
    scale *= params_.resize_factor;
    min_scale *= params_.resize_factor;
  }
  return scales;
}

void MTCNNImpl::Preprocess(const cv::Mat& img,
                       std::vector<cv::Mat>* input_channels) {
  cv::Mat sample;
  img.convertTo(sample, CV_32FC3, params_.normalize_factor,
                -params_.mean_value * params_.normalize_factor);
  cv::split(sample, *input_channels);
}

void MTCNNImpl::NetInference(const std::vector<cv::Mat>& imgs, const int stage) {
  caffe::Blob<float>* input_layer =
    nets_[stage - 1]->blob_by_name(params_.input_name).get();
  if (stage == 1) {
    input_geometry_[0].width = imgs.front().cols;
    input_geometry_[0].height = imgs.front().rows;
  }
  input_layer->Reshape(imgs.size(), num_channels_[stage - 1],
                       input_geometry_[stage - 1].height,
                       input_geometry_[stage - 1].width);
  nets_[stage - 1]->Reshape();

  for (int i = 0; i < imgs.size(); ++i) {
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(stage, i, &input_channels);
    Preprocess(imgs[i], &input_channels);
  }
  nets_[stage - 1]->Forward();
}

void MTCNNImpl::PNetProcess(const cv::Mat& img, std::vector<BBox>& bboxes) {
  std::vector<float> scales = PyramidImage(img.size());
  for (auto& scale : scales) {
    int width = std::ceil(img.cols * scale + 1.0f);
    int height = std::ceil(img.rows * scale + 1.0f);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(width, height), 0, 0, cv::INTER_AREA);

    NetInference(std::vector<cv::Mat>(1, resized), 1);

    caffe::Blob<float>* offset_layer =
      nets_[0]->blob_by_name(params_.offset_name[0]).get();
    caffe::Blob<float>* prob_layer =
      nets_[0]->blob_by_name(params_.prob_name).get();
    const float* offset = offset_layer->cpu_data();
    const float* prob = prob_layer->cpu_data();

    std::vector<BBox> candidates =
      GenerateBBox(offset, prob, prob_layer->num(),
                          prob_layer->width(), prob_layer->height(),
                          params_.stride_size, params_.cell_size,
                          params_.confidence_thresh[0], scale)
      .front();
    for (auto& bbox : candidates) {
      std::swap(bbox.offset.dx1, bbox.offset.dy1);
      std::swap(bbox.offset.dx2, bbox.offset.dy2);
    }
    BBoxNMS(candidates, params_.nms_thresh1, Union);
    bboxes.insert(bboxes.end(), candidates.begin(), candidates.end());
  }
  BBoxNMS(bboxes, params_.nms_thresh, Union);
  BBoxRegress(bboxes, 0.0f);
}

void MTCNNImpl::RNetProcess(const cv::Mat& img, std::vector<BBox>& bboxes) {
  BBox2Square(bboxes);
  std::vector<cv::Mat> samples;
  cv::Rect rect;
#ifdef CPU_ONLY
  std::vector<float> offset_buf(bboxes.size() * 4);
  std::vector<float> prob_buf(bboxes.size() * 2);
#endif
  for(size_t i = 0; i < bboxes.size(); ++i) {
    // left, right, top, bottom
    cv::Vec4i padding = BBoxPadding(bboxes[i], img.size(), rect);
    cv::Mat sample = img(rect);
    cv::copyMakeBorder(sample, sample, padding[2], padding[3], padding[0],
                       padding[1], cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::resize(sample, sample, input_geometry_[1], 0, 0, cv::INTER_AREA);
    samples.push_back(sample);
#ifdef CPU_ONLY
    NetInference(samples, 2);
    const float* offset = nets_[1]->blob_by_name(params_.offset_name[1]).get()->cpu_data();
    const float* prob = nets_[1]->blob_by_name(params_.prob_name).get()->cpu_data();
    memcpy(offset_buf.data() + i * 4, offset, 4 * sizeof(float));
    memcpy(prob_buf.data() + i * 2, prob, 2 * sizeof(float));
    samples.clear();
#endif
  }
#ifndef CPU_ONLY
  NetInference(samples, 2);

  caffe::Blob<float>* offset_layer =
    nets_[1]->blob_by_name(params_.offset_name[1]).get();
  caffe::Blob<float>* prob_layer =
    nets_[1]->blob_by_name(params_.prob_name).get();
  const float* offset = offset_layer->cpu_data();
  const float* prob = prob_layer->cpu_data();
#else
  const float* offset = offset_buf.data();
  const float* prob = prob_buf.data();
#endif
  std::vector<BBox> candidates;
  for (int i = 0; i < bboxes.size(); ++i) {
    if (prob[i * 2 + 1] >= params_.confidence_thresh[1]) {
      bboxes[i].score = prob[i * 2 + 1];
      bboxes[i].offset.dy1 = offset[i * 4 + 0];  // std::swap
      bboxes[i].offset.dx1 = offset[i * 4 + 1];
      bboxes[i].offset.dy2 = offset[i * 4 + 2];
      bboxes[i].offset.dx2 = offset[i * 4 + 3];
      candidates.push_back(std::move(bboxes[i]));
    }
  }
  std::swap(bboxes, candidates);
  BBoxNMS(bboxes, params_.nms_thresh, Union);
  BBoxRegress(bboxes, 1.0f);
}

void MTCNNImpl::ONetProcess(const cv::Mat& img, std::vector<BBox>& bboxes) {
  BBox2Square(bboxes);
  std::vector<cv::Mat> samples;
  cv::Rect rect;
#ifdef CPU_ONLY
  std::vector<float> offset_buf(bboxes.size() * 4);
  std::vector<float> prob_buf(bboxes.size() * 2);
  std::vector<float> landmakrs_buf(bboxes.size() * params_.landmark_num * 2);
#endif
  for (size_t i = 0; i < bboxes.size(); ++i) {
    cv::Vec4i padding = BBoxPadding(bboxes[i], img.size(), rect);
    cv::Mat sample = img(rect);
    cv::copyMakeBorder(sample, sample, padding[2], padding[3], padding[0],
                       padding[1], cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::resize(sample, sample, input_geometry_[2], 0, 0, cv::INTER_AREA);
    samples.push_back(sample);
#ifdef CPU_ONLY
    NetInference(samples, 3);
    const float* offset = nets_[2]->blob_by_name(params_.offset_name[2]).get()->cpu_data();
    const float* prob = nets_[2]->blob_by_name(params_.prob_name).get()->cpu_data();
    const float* landmark = nets_[2]->blob_by_name(params_.landmark_name).get()->cpu_data();
    memcpy(offset_buf.data() + i * 4, offset, 4 * sizeof(float));
    memcpy(prob_buf.data() + i * 2, prob, 2 * sizeof(float));
    memcpy(landmakrs_buf.data() + i * params_.landmark_num * 2, 
           landmark, params_.landmark_num * 2 * sizeof(float));
    samples.clear();
#endif
  }
#ifndef CPU_ONLY
  NetInference(samples, 3);
  caffe::Blob<float>* offset_layer =
    nets_[2]->blob_by_name(params_.offset_name[2]).get();
  caffe::Blob<float>* prob_layer =
    nets_[2]->blob_by_name(params_.prob_name).get();
  caffe::Blob<float>* landmark_layer =
    nets_[2]->blob_by_name(params_.landmark_name).get();
  const float* offset = offset_layer->cpu_data();
  const float* prob = prob_layer->cpu_data();
  const float* landmark = landmark_layer->cpu_data();
#else
  const float* offset = offset_buf.data();
  const float* prob = prob_buf.data();
  const float* landmark = landmakrs_buf.data();
#endif

  std::vector<BBox> candidates;
  for (int i = 0; i < bboxes.size(); ++i) {
    if (prob[i * 2 + 1] >= params_.confidence_thresh[2]) {
      bboxes[i].score = prob[i * 2 + 1];
      bboxes[i].offset.dy1 = offset[i * 4 + 0];  // std::swap
      bboxes[i].offset.dx1 = offset[i * 4 + 1];
      bboxes[i].offset.dy2 = offset[i * 4 + 2];
      bboxes[i].offset.dx2 = offset[i * 4 + 3];

      std::vector<Landmark> landmarks(params_.landmark_num, Landmark());
      // x1, x2, x3, x4, x5, y1, y2, y3, y4, y5 for MATLAB
      for (int j = 0; j < params_.landmark_num; ++j) {
        landmarks[j].x =
          bboxes[i].tl.x + landmark[j + 5] * bboxes[i].height() - 1;
        landmarks[j].y = bboxes[i].tl.y + landmark[j] * bboxes[i].width() - 1;
      }
      bboxes[i].landmarks.swap(landmarks);
      candidates.push_back(std::move(bboxes[i]));
    }
    landmark += params_.landmark_num * 2;
  }
  std::swap(bboxes, candidates);
  BBoxRegress(bboxes, 1.0f);
  BBoxNMS(bboxes, params_.nms_thresh, Min);
}
