#include "yolodetector.h"
#include "OCSortTracker.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <string>
#include <fcntl.h>



static std::streambuf* original_cout = nullptr;
static std::streambuf* original_cerr = nullptr;

#ifdef _WIN32
    #include <io.h>
    #define DUP2 _dup2
    #define FILENO _fileno
    #define NULL_PATH "NUL"
#else
    #include <unistd.h>
    #define DUP2 dup2
    #define FILENO fileno
    #define NULL_PATH "/dev/null"
#endif

void disable_logging() {

    freopen(NULL_PATH, "w", stderr);
    freopen(NULL_PATH, "w", stdout);

    int null_fd = open(NULL_PATH, O_WRONLY);
    DUP2(null_fd, FILENO(stdout));
    DUP2(null_fd, FILENO(stderr));

    original_cout = std::cout.rdbuf();
    original_cerr = std::cerr.rdbuf();

    static std::ofstream null_stream;
    null_stream.open(NULL_PATH);
    std::cout.rdbuf(null_stream.rdbuf());
    std::cerr.rdbuf(null_stream.rdbuf());
}

void enable_logging() {

    #ifdef _WIN32
        freopen("CON", "w", stdout);
        freopen("CON", "w", stderr);
    #else
        freopen("/dev/tty", "w", stdout);
        freopen("/dev/tty", "w", stderr);
    #endif

    if (original_cout) {
        std::cout.rdbuf(original_cout);
    }
    if (original_cerr) {
        std::cerr.rdbuf(original_cerr);
    }
}


void clear_terminal() {
#ifdef _WIN32
    std::system("cls");
#else
    std::system("clear");
#endif
}

const double FONT_SCALE = 0.7;
const int THICKNESS = 1;
const cv::Scalar BLACK = cv::Scalar(0, 0, 0);
const cv::Scalar BLUE = cv::Scalar(255, 178, 50);
const cv::Scalar RED = cv::Scalar(0, 0, 255);
const cv::Scalar YELLOW = cv::Scalar(0, 255, 255);

void draw_label(cv::Mat &im, const std::string &label, int x, int y) {
    int baseline;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS, &baseline);
    cv::rectangle(im, cv::Point(x, y), cv::Point(x + text_size.width, y + text_size.height + baseline),
                  BLACK, cv::FILLED);
    cv::putText(im, label, cv::Point(x, y + text_size.height), cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, YELLOW, THICKNESS,
                cv::LINE_AA);
}

cv::Mat visualize_detections_and_tracks(
    cv::Mat &input_image,
    const std::vector<YoloDetector::Detection>& detections,
    const std::vector<TrackingResult>& tracks,
    const std::vector<std::string>& classes,
    const std::map<int, int>* track_to_super_id = nullptr) {

    for (const auto& det : detections) {
        cv::Rect box(
            static_cast<int>(det.x1),
            static_cast<int>(det.y1),
            static_cast<int>(det.x2 - det.x1),
            static_cast<int>(det.y2 - det.y1)
        );

        cv::rectangle(input_image, box, BLUE, 3 * THICKNESS);
        std::string label = cv::format("%s: %.2f", classes[det.class_id].c_str(), det.confidence);
        draw_label(input_image, label, box.x, box.y);
    }

    for (const auto& track : tracks) {
        cv::Rect track_box(
            static_cast<int>(track.x1),
            static_cast<int>(track.y1),
            static_cast<int>(track.x2 - track.x1),
            static_cast<int>(track.y2 - track.y1)
        );

        cv::rectangle(input_image, track_box, RED, THICKNESS);

        std::string track_label;
        if (track_to_super_id && track_to_super_id->count(track.track_id)) {
            int super_id = track_to_super_id->at(track.track_id);
            track_label = cv::format("ID:%d (S:%d)", track.track_id, super_id);
        } else {
            track_label = cv::format("ID:%d", track.track_id);
        }

        cv::putText(input_image, track_label,
                   cv::Point(track.x1, track.y1 - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv::LINE_AA);
    }

    return input_image;
}

std::string get_camera_info() {
    std::stringstream info;
    const std::vector<std::tuple<int, int, int>> test_configs = {
        {3840, 2160, 30},
        {3840, 2160, 60},
        {1920, 1080, 30},
        {1920, 1080, 60},
        {1280, 720, 30},
        {1280, 720, 60}
    };

    try {
        for (int i = 0; i < 10; i++) {
            cv::VideoCapture cap;

            #ifdef _WIN32
            cap.open(i, cv::CAP_DSHOW);
            #else
            cap.open(i);
            #endif

            if (!cap.isOpened()) continue;
            const std::string backend = cap.getBackendName();
            cap.release();

            info << "Camera " << i << ":\n";
            info << "  Backend: " << backend << "\n";
            info << "  Supported modes:\n";

            std::set<std::string> unique_modes;

            for (const auto& [width, height, target_fps] : test_configs) {
                cv::VideoCapture test_cap;
                #ifdef _WIN32
                test_cap.open(i, cv::CAP_DSHOW);
                #else
                test_cap.open(i);
                #endif

                if (!test_cap.isOpened()) continue;

                test_cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
                test_cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
                test_cap.set(cv::CAP_PROP_FPS, target_fps);

                double actual_w = test_cap.get(cv::CAP_PROP_FRAME_WIDTH);
                double actual_h = test_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
                double actual_fps = test_cap.get(cv::CAP_PROP_FPS);
                test_cap.release();

                if (static_cast<int>(actual_w) == width &&
                    static_cast<int>(actual_h) == height) {
                    std::string mode = std::to_string(width) + "x" + std::to_string(height);
                    std::string fps_str = (actual_fps > 0)
                                        ? std::to_string(static_cast<int>(actual_fps)) + " fps"
                                        : "unknown fps";

                    bool exists = false;
                    for (const auto& um : unique_modes) {
                        if (um.find(mode) != std::string::npos) {
                            exists = true;
                            break;
                        }
                    }

                    if (!exists) {
                        unique_modes.insert(mode + " @ " + fps_str);
                    }
                }
            }

            for (const auto& mode : unique_modes) {
                info << "    - " << mode << "\n";
            }
            info << "\n";
        }
    }
    catch (const cv::Exception& e) {
        info << "OpenCV error: " << e.what() << "\n";
    }
    catch (const std::exception& e) {
        info << "Error: " << e.what() << "\n";
    }

    return info.str();
}


int oneCameraProto() {


    disable_logging();

     // std::string camera_info = get_camera_info();
    std::cout << "loading YOLO network... (pre-clear)" << std::endl;

    enable_logging();
     //std::cout << "Found cameras:\n" << camera_info << std::endl;

    std::cout << "loading YOLO network... (post-clear)" << std::endl;


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
        YoloDetector detector("assets/yolov9-m.onnx");
        std::cout << "network loaded successfully" << std::endl;

        OCSortTracker tracker;
        std::cout << "tracker initialized" << std::endl;

        std::cout << "opening camera" << std::endl;
        cv::VideoCapture cap(2);



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

            auto detections = detector.detect(frame);

            auto tracks = tracker.update(detections);

            cv::Mat img = visualize_detections_and_tracks(frame, detections, tracks, classes);

            cv::imshow("YOLO V9 with Tracking", img);

            if (cv::waitKey(1) == 'q') {
                std::cout << "quit signal received" << std::endl;
                break;
            }
        }

        cap.release();
        cv::destroyAllWindows();

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "unexpected error: " << e.what() << std::endl;
        return -1;
    }
}