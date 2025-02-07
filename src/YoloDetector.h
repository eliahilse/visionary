#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include <vector>

class YoloDetector {
public:
    struct Detection {
        float x1, y1, x2, y2;
        float confidence;
        int class_id;
    };

    explicit YoloDetector(const std::string& model_path,
                 float conf_threshold = 0.4, 
                 float nms_threshold = 0.4);

    std::vector<Detection> detect(const cv::Mat& input_image);

private:
    cv::dnn::Net net;
    const float CONFIDENCE_THRESHOLD;
    const float NMS_THRESHOLD;
    static constexpr int INPUT_WIDTH = 640;
    static constexpr int INPUT_HEIGHT = 640;

    void setBestRuntime(cv::dnn::Net& net);
    cv::Mat preProcess(const cv::Mat& input_image);
    std::vector<Detection> postProcess(const cv::Mat& input_image, const cv::Mat& output);
};