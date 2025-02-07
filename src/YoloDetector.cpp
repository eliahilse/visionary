#include "YoloDetector.h"
#include <stdexcept>

namespace {
    bool checkCudaSupport() {
        int device_count = cv::cuda::getCudaEnabledDeviceCount();

        #ifndef NDEBUG
                std::cout << "Number of CUDA devices: " << device_count << std::endl;
        #endif

        if (device_count > 0) {
            for (int i = 0; i < device_count; ++i) {
                cv::cuda::printShortCudaDeviceInfo(i);
            }
            return true;
        } else {

        #ifndef NDEBUG
                    std::cout << "No CUDA-enabled devices found." << std::endl;
        #endif

        return false;
        }
    }

    bool checkOpenCLsupport() {
        return cv::ocl::haveOpenCL();
    }
}

YoloDetector::YoloDetector(const std::string& model_path, 
                          float conf_threshold, 
                          float nms_threshold) 
    : CONFIDENCE_THRESHOLD(conf_threshold)
    , NMS_THRESHOLD(nms_threshold) {
    try {
        net = cv::dnn::readNet(model_path);
        setBestRuntime(net);
    } catch (const cv::Exception& e) {
        throw std::runtime_error("Failed to load network: " + std::string(e.what()));
    }
}

void YoloDetector::setBestRuntime(cv::dnn::Net& net) {
    if (checkCudaSupport()) {
        std::cout << "Utilizing CUDA Runtime" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else if (checkOpenCLsupport()) {
        // todo: check if working properly
        std::cout << "No CUDA device found, fallback to OpenCL" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    } else {
        std::cout << "No CUDA / OpenCL device found, fallback to raw CPU" << std::endl;
    }
}


cv::Mat YoloDetector::preProcess(const cv::Mat& input_image) {
    cv::Mat blob;
    cv::dnn::blobFromImage(input_image, blob, 1./255., 
                          cv::Size(INPUT_WIDTH, INPUT_HEIGHT), 
                          cv::Scalar(), true, false);
    net.setInput(blob);
    return blob;
}

std::vector<YoloDetector::Detection> YoloDetector::detect(const cv::Mat& input_image) {
    cv::Mat blob = preProcess(input_image);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    
    return postProcess(input_image, outputs[0]);
}

std::vector<YoloDetector::Detection> YoloDetector::postProcess(
    const cv::Mat& input_image, const cv::Mat& output) {
    
    cv::Mat reshaped_output = output.reshape(1, 84);
    cv::Mat transposed_output;
    cv::transpose(reshaped_output, transposed_output);

    cv::Mat boxes = transposed_output.colRange(0, 4);
    cv::Mat scores = transposed_output.colRange(4, transposed_output.cols);

    std::vector<cv::Point2f> points;
    std::vector<int> class_ids;
    std::vector<float> confidences;

    for (int i = 0; i < scores.rows; ++i) {
        cv::Mat scores_row = scores.row(i);
        cv::Point class_id_point;
        double confidence;
        cv::minMaxLoc(scores_row, nullptr, &confidence, nullptr, &class_id_point);

        if (confidence > CONFIDENCE_THRESHOLD) {
            cv::Mat box = boxes.row(i);
            float x = box.at<float>(0);
            float y = box.at<float>(1);
            float w = box.at<float>(2);
            float h = box.at<float>(3);

            float x1 = (x - w/2) * input_image.cols / INPUT_WIDTH;
            float y1 = (y - h/2) * input_image.rows / INPUT_HEIGHT;
            float x2 = (x + w/2) * input_image.cols / INPUT_WIDTH;
            float y2 = (y + h/2) * input_image.rows / INPUT_HEIGHT;

            points.push_back(cv::Point2f(x1, y1));
            points.push_back(cv::Point2f(x2, y2));
            class_ids.push_back(class_id_point.x);
            confidences.push_back(static_cast<float>(confidence));
        }
    }

    std::vector<int> indices;
    std::vector<cv::Rect2d> boxes_nms;
    for (size_t i = 0; i < points.size(); i += 2) {
        boxes_nms.push_back(cv::Rect2d(
            points[i].x, points[i].y,
            points[i+1].x - points[i].x,
            points[i+1].y - points[i].y
        ));
    }

    cv::dnn::NMSBoxes(boxes_nms, confidences, CONFIDENCE_THRESHOLD, 
                      NMS_THRESHOLD, indices);

    std::vector<Detection> detections;
    for (int idx : indices) {
        Detection det;
        det.x1 = points[idx*2].x;
        det.y1 = points[idx*2].y;
        det.x2 = points[idx*2+1].x;
        det.y2 = points[idx*2+1].y;
        det.confidence = confidences[idx];
        det.class_id = class_ids[idx];
        detections.push_back(det);
    }

    return detections;
}