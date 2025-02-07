#include "OCSortTracker.h"

OCSortTracker::OCSortTracker(float delta_t, 
                           int max_age,
                           int min_hits,
                           float iou_threshold,
                           int associate_method,
                           const std::string& distance_metric,
                           float inertia,
                           bool use_byte)
    : tracker_(delta_t, max_age, min_hits, iou_threshold, 
              associate_method, distance_metric, inertia, use_byte) {}

std::vector<TrackingResult> OCSortTracker::update(const std::vector<YoloDetector::Detection>& detections) {
    std::vector<std::vector<float>> detection_data;
    
    for (const auto& det : detections) {
        std::vector<float> row{
            det.x1, det.y1, det.x2, det.y2,
            det.confidence,
            static_cast<float>(det.class_id)
        };
        detection_data.push_back(row);
    }

    if (detection_data.empty()) {
        return {};
    }

    auto matrix = Vector2Matrix(detection_data);
    std::vector<Eigen::RowVectorXf> tracking_output = tracker_.update(matrix);

    std::vector<TrackingResult> results;
    for (const auto& track : tracking_output) {
        TrackingResult result{
            track[0],
            track[1],
            track[2],
            track[3],
            static_cast<int>(track[4]),
            static_cast<int>(track[5]),
            track[6]
        };
        results.push_back(result);
    }

    return results;
}

Eigen::Matrix<float, Eigen::Dynamic, 6> OCSortTracker::Vector2Matrix(const std::vector<std::vector<float>>& data) {
    if (data.empty()) return Eigen::Matrix<float, Eigen::Dynamic, 6>(0, 6);
    
    Eigen::Matrix<float, Eigen::Dynamic, 6> matrix(data.size(), 6);
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < 6; ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    return matrix;
}