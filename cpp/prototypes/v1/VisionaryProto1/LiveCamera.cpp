#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <vector>



// consts
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

// text settings
const double FONT_SCALE = 0.7;
const int THICKNESS = 1;

// box colors
const cv::Scalar BLACK = cv::Scalar(0, 0, 0);
const cv::Scalar BLUE = cv::Scalar(255, 178, 50);
const cv::Scalar YELLOW = cv::Scalar(0, 255, 255);

void draw_label(cv::Mat& im, const std::string& label, int x, int y) {
    int baseline;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS, &baseline);
    cv::rectangle(im, cv::Point(x, y), cv::Point(x + text_size.width, y + text_size.height + baseline),
        BLACK, cv::FILLED);
    cv::putText(im, label, cv::Point(x, y + text_size.height), cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, YELLOW, THICKNESS, cv::LINE_AA);
}


cv::Mat pre_process(const cv::Mat& input_image, cv::dnn::Net& net) {

    // transform img to fixed model w/h +/ blob
    cv::Mat blob;
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);

    // submit to model
    net.setInput(blob);

    // get model detections
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs[0];
}

cv::Mat post_process(cv::Mat& input_image, const cv::Mat& output, const std::vector<std::string>& classes) {
    std::cout << "Output shape: " << output.size << std::endl;
    std::cout << "Output type: " << cv::typeToString(output.type()) << std::endl;
    std::cout << "Output total: " << output.total() << std::endl;

    // transpose output // fix format
    cv::Mat reshaped_output = output.reshape(1, 84);
    cv::Mat transposed_output;
    cv::transpose(reshaped_output, transposed_output);

    // separate boxes & confidence scores
    cv::Mat boxes = transposed_output.colRange(0, 4);
    cv::Mat scores = transposed_output.colRange(4, transposed_output.cols);

    std::vector<cv::Rect> output_boxes;
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

            // cornder coords
            int left = static_cast<int>((x - w / 2) * input_image.cols / INPUT_WIDTH);
            int top = static_cast<int>((y - h / 2) * input_image.rows / INPUT_HEIGHT);
            int width = static_cast<int>(w * input_image.cols / INPUT_WIDTH);
            int height = static_cast<int>(h * input_image.rows / INPUT_HEIGHT);

            output_boxes.push_back(cv::Rect(left, top, width, height));
            class_ids.push_back(class_id_point.x);
            confidences.push_back(static_cast<float>(confidence));
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(output_boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

    for (int idx : indices) {
        cv::Rect box = output_boxes[idx];
        cv::rectangle(input_image, box, BLUE, 3 * THICKNESS);
        std::string label = cv::format("%s: %.2f", classes[class_ids[idx]].c_str(), confidences[idx]);
        draw_label(input_image, label, box.x, box.y);

        std::cout << "! detection: ";
        std::cout << "class=" << classes[class_ids[idx]] << ", confidence=" << confidences[idx] << ", ";
        std::cout << "box=" << box.x << "," << box.y << "," << box.width << "," << box.height << std::endl;
    }

    return input_image;
}


int liveCameraProto() {
    int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "cuda on? " << (cuda_devices > 0 ? "y" : "n") << std::endl;
    try {
        std::vector<std::string> classes;
        std::ifstream ifs("assets/coco.names");
        if (!ifs.is_open()) {
            std::cerr << "Error opening coco.names file" << std::endl;
            return -1;
        }
        std::string line;
        while (getline(ifs, line)) {
            classes.push_back(line);
        }

        std::cout << "loading YOLO network..." << std::endl;
        cv::dnn::Net net;
        try {
            net = cv::dnn::readNet("assets/yolov8m.onnx");
        }
        catch (const cv::Exception& e) {
            std::cerr << "error loading network: " << e.what() << std::endl;
            return -1;
        }
        std::cout << "network loaded successfully" << std::endl;

        std::cout << "opening camera" << std::endl;
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "error! -> perm denied" << std::endl;
            return -1;
        }
        std::cout << "camera open success" << std::endl;

        while (true) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "! error capturing frame" << std::endl;
                break;
            }

            cv::Mat detections = pre_process(frame, net);

            cv::Mat img = post_process(frame, detections, classes);

            cv::imshow("yolo v8", img);

            if (cv::waitKey(1) == 'q') {
                std::cout << "quit signal received" << std::endl;
                break;
            }
        }

        cap.release();
        cv::destroyAllWindows();

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "unexpected error: " << e.what() << std::endl;
        return -1;
    }
}