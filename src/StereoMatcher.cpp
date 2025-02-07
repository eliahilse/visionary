#include "StereoMatcher.h"
#include <algorithm>

StereoMatcher::StereoMatcher(float image_width)
    : img_width(image_width) {}

std::vector<std::vector<int>> StereoMatcher::computeCostMatrix(
    const std::vector<TrackingResult>& left_tracks,
    const std::vector<TrackingResult>& right_tracks
) {
    const size_t left_count = left_tracks.size();
    const size_t right_count = right_tracks.size();

    std::vector<std::vector<int>> cost_matrix(left_count,
                                             std::vector<int>(right_count, 0));

    for (size_t i = 0; i < left_count; i++) {
        const auto& left_track = left_tracks[i];
        cv::Point2f left_center = computeBoxCenter(left_track);
        float left_area = computeBoxArea(left_track);

        for (size_t j = 0; j < right_count; j++) {
            const auto& right_track = right_tracks[j];
            cv::Point2f right_center = computeBoxCenter(right_track);
            float right_area = computeBoxArea(right_track);

            float cost = 0.0f;

            float vert_diff = std::abs(right_center.y - left_center.y);
            cost += VERTICAL_WEIGHT * vert_diff;

            float horiz_diff = right_center.x - left_center.x;
            if (horiz_diff < 0) {
                cost += NEGATIVE_DISP_PENALTY * std::abs(horiz_diff);
            } else {
                cost += HORIZONTAL_WEIGHT * horiz_diff;
            }

            float area_diff = std::abs(right_area - left_area) /
                            std::max(right_area, left_area);
            cost += SIZE_WEIGHT * area_diff;

            if (left_track.class_id != right_track.class_id) {
                cost += CLASS_MISMATCH_PENALTY;
            }

            cost_matrix[i][j] = static_cast<int>(cost * 100.0f);
        }
    }

    return cost_matrix;
}

std::vector<StereoPair> StereoMatcher::greedyMatch(
    const std::vector<std::vector<int>>& cost_matrix,
    const std::vector<TrackingResult>& left_tracks,
    const std::vector<TrackingResult>& right_tracks
) {
    std::vector<StereoPair> matches;
    if (cost_matrix.empty()) return matches;

    std::vector<bool> left_matched(left_tracks.size(), false);
    std::vector<bool> right_matched(right_tracks.size(), false);

    while (true) {
        int min_cost = static_cast<int>(CLASS_MISMATCH_PENALTY * 100.0f);
        int best_left = -1;
        int best_right = -1;

        for (size_t i = 0; i < left_tracks.size(); i++) {
            if (left_matched[i]) continue;

            for (size_t j = 0; j < right_tracks.size(); j++) {
                if (right_matched[j]) continue;

                if (cost_matrix[i][j] < min_cost) {
                    min_cost = cost_matrix[i][j];
                    best_left = i;
                    best_right = j;
                }
            }
        }

        if (best_left == -1 || best_right == -1) break;

        StereoPair pair;
        pair.left_id = left_tracks[best_left].track_id;
        pair.right_id = right_tracks[best_right].track_id;
        matches.push_back(pair);

        left_matched[best_left] = true;
        right_matched[best_right] = true;
    }

    return matches;
}


std::vector<StereoPair> StereoMatcher::matchTracks(
    const std::vector<TrackingResult>& left_tracks,
    const std::vector<TrackingResult>& right_tracks
) {
    if (left_tracks.empty() || right_tracks.empty()) {
        return std::vector<StereoPair>();
    }

     auto cost_matrix = computeCostMatrix(left_tracks, right_tracks);
    return greedyMatch(cost_matrix, left_tracks, right_tracks);

    /* hungarian instead of (not really suitable) greedy match -- todo @ elia */
}


float StereoMatcher::computeBoxArea(const TrackingResult& track) {
    float width = track.x2 - track.x1;
    float height = track.y2 - track.y1;
    return width * height;
}

cv::Point2f StereoMatcher::computeBoxCenter(const TrackingResult& track) {
    return cv::Point2f(
        (track.x1 + track.x2) / 2.0f,
        (track.y1 + track.y2) / 2.0f
    );
}