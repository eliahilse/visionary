#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <list>
#include "OCSortTracker.h"

struct StereoPair {
    int left_id;
    int right_id;
};

class StereoMatcher {
public:
    explicit StereoMatcher(float image_width = 640.0f);

    std::vector<StereoPair> matchTracks(
        const std::vector<TrackingResult>& left_tracks,
        const std::vector<TrackingResult>& right_tracks
    );

private:
    float img_width;

    static constexpr float VERTICAL_WEIGHT = 5.0f;
    static constexpr float HORIZONTAL_WEIGHT = 1.0f;
    static constexpr float NEGATIVE_DISP_PENALTY = 10.0f;
    static constexpr float SIZE_WEIGHT = 2.0f;
    static constexpr float CLASS_MISMATCH_PENALTY = 150.0f;

    std::vector<std::vector<int>> computeCostMatrix(
        const std::vector<TrackingResult>& left_tracks,
        const std::vector<TrackingResult>& right_tracks
    );

    static float computeBoxArea(const TrackingResult& track);
    static cv::Point2f computeBoxCenter(const TrackingResult& track);

    std::vector<StereoPair> greedyMatch(const std::vector<std::vector<int>>& cost_matrix,
                                       const std::vector<TrackingResult>& left_tracks,
                                       const std::vector<TrackingResult>& right_tracks);

};