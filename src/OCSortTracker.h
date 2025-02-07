#pragma once

#include <Eigen/Dense>
#include <vector>
#include <opencv2/core/mat.hpp>
#include "YoloDetector.h"
#include "../../oc-sort/deploy/OCSort/cpp/include/OCSort.hpp"

struct TrackingResult {
    float x1, y1, x2, y2;
    int track_id;
    int class_id;
    float confidence;
};

class OCSortTracker {
public:
    OCSortTracker(float delta_t = 0.1,
                  int max_age = 50,
                  int min_hits = 1,
                  float iou_threshold = 0.22,
                  int associate_method = 1,
                  const std::string& distance_metric = "giou",
                  float inertia = 0.3941737016672115,
                  bool use_byte = true);

    std::vector<TrackingResult> update(const std::vector<YoloDetector::Detection>& detections);

private:
    ocsort::OCSort tracker_;

    static Eigen::Matrix<float, Eigen::Dynamic, 6> Vector2Matrix(const std::vector<std::vector<float>>& data);
};