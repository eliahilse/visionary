#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include "YoloDetector.h"

int oneCameraProto();


void disable_logging();
void enable_logging();
void clear_terminal();
void draw_label(cv::Mat &im, const std::string &label, int x, int y);
cv::Mat visualize_detections_and_tracks(
    cv::Mat &input_image,
    const std::vector<YoloDetector::Detection>& detections,
    const std::vector<TrackingResult>& tracks,
    const std::vector<std::string>& classes,
    const std::map<int, int>* track_to_super_id = nullptr
);
std::string get_camera_info();