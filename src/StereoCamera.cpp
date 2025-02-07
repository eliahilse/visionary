#include "YoloDetector.h"
#include "OCSortTracker.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <fcntl.h>
#include "OneCamera.h"
#include "StereoMatcher.h"


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




#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

struct CameraBuffer {
    cv::Mat frame;
    std::mutex mutex;
    bool has_new_frame = false;
    std::atomic<bool> stop{false};
};

std::vector<int> showCameraGrid() {
    std::vector<cv::VideoCapture> caps;
    std::vector<int> valid_indices;
    std::vector<std::shared_ptr<CameraBuffer>> buffers;
    std::vector<std::thread> capture_threads;

    for(int i = 0; i < 10; i++) {
        cv::VideoCapture cap(i);
        if(cap.isOpened()) {
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            cap.set(cv::CAP_PROP_FPS, 30);

            cv::Mat test_frame;
            cap.read(test_frame);
            if(!test_frame.empty()) {
                valid_indices.push_back(i);
                caps.push_back(std::move(cap));
                buffers.push_back(std::make_shared<CameraBuffer>());
            }
        }
    }

    if(caps.empty()) return valid_indices;

    // start capture threads
    for(size_t i = 0; i < caps.size(); i++) {
        capture_threads.emplace_back([&caps, &buffers, i]() {
            while(!buffers[i]->stop) {
                cv::Mat new_frame;
                if(caps[i].read(new_frame) && !new_frame.empty()) {
                    std::lock_guard<std::mutex> lock(buffers[i]->mutex);
                    new_frame.copyTo(buffers[i]->frame);
                    buffers[i]->has_new_frame = true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
            }
        });
    }

    cv::namedWindow("Cameras Overview - Press Any Key to Continue", cv::WINDOW_NORMAL);

    const int grid_cols = 3;
    std::vector<cv::Mat> display_frames(caps.size());
    bool first_frame = true;

    while(true) {
        bool any_new_frames = false;

        // update frames
        for(size_t i = 0; i < buffers.size(); i++) {
            std::lock_guard<std::mutex> lock(buffers[i]->mutex);
            if(buffers[i]->has_new_frame) {
                buffers[i]->frame.copyTo(display_frames[i]);
                cv::putText(display_frames[i], std::to_string(valid_indices[i]),
                           cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                           1.0, cv::Scalar(0, 255, 0), 2);
                buffers[i]->has_new_frame = false;
                any_new_frames = true;
            }
        }

        // update only for first frames / new frames
        if(any_new_frames || first_frame) {
            std::vector<cv::Mat> grid_rows;
            std::vector<cv::Mat> current_row;

            for(const auto& frame : display_frames) {
                if(!frame.empty()) {
                    current_row.push_back(frame);
                } else {
                    current_row.push_back(cv::Mat::zeros(480, 640, CV_8UC3));
                }

                if(current_row.size() == grid_cols) {
                    cv::Mat row;
                    cv::hconcat(current_row, row);
                    grid_rows.push_back(row);
                    current_row.clear();
                }
            }

            if(!current_row.empty()) {
                while(current_row.size() < grid_cols) {
                    current_row.push_back(cv::Mat::zeros(480, 640, CV_8UC3));
                }
                cv::Mat row;
                cv::hconcat(current_row, row);
                grid_rows.push_back(row);
            }

            if(!grid_rows.empty()) {
                cv::Mat grid;
                cv::vconcat(grid_rows, grid);
                cv::imshow("Cameras", grid);
                first_frame = false;
            }
        }

        int key = cv::waitKey(1);
        if(key >= 0) break;
    }

    for(auto& buffer : buffers) {
        buffer->stop = true;
    }

    for(auto& thread : capture_threads) {
        if(thread.joinable()) {
            thread.join();
        }
    }

    for(auto& cap : caps) {
        cap.release();
    }

    cv::destroyAllWindows();
    return valid_indices;
}

struct CameraProcessor {
    cv::Mat frame;
    std::vector<YoloDetector::Detection> detections;
    std::vector<TrackingResult> tracks;
    std::mutex mutex;
    std::atomic<bool> has_new_frame{false};
    std::atomic<bool> stop{false};
};

int stereoCameraProto() {
    std::vector<int> available_cams = showCameraGrid();
    if(available_cams.empty()) {
        std::cout << "No cameras found!" << std::endl;
        return -1;
    }

    int left_idx, right_idx;
    std::cout << "\nAvailable cameras: ";
    for(int cam : available_cams) std::cout << cam << " ";
    std::cout << "\nEnter left camera index: ";
    std::cin >> left_idx;
    std::cout << "Enter right camera index: ";
    std::cin >> right_idx;

    std::vector<std::string> classes;
    std::ifstream ifs("assets/coco.names");
    std::string line;
    while(getline(ifs, line)) classes.push_back(line);

    auto left_detector = std::make_shared<YoloDetector>("assets/yolov9-m.onnx");
    auto right_detector = std::make_shared<YoloDetector>("assets/yolov9-m.onnx");
    auto left_processor = std::make_shared<CameraProcessor>();
    auto right_processor = std::make_shared<CameraProcessor>();

    OCSortTracker left_tracker, right_tracker;

    auto process_camera = [](int camera_idx,
                           std::shared_ptr<CameraProcessor> processor,
                           std::shared_ptr<YoloDetector> detector,
                           OCSortTracker& tracker) {
        cv::VideoCapture cap(camera_idx);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open camera " << camera_idx << std::endl;
            return;
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 640);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        cap.set(cv::CAP_PROP_FPS, 30);

        cv::Mat frame;
        while(!processor->stop) {
            if(cap.read(frame) && !frame.empty()) {
                auto detections = detector->detect(frame);
                auto tracks = tracker.update(detections);

                std::lock_guard<std::mutex> lock(processor->mutex);
                frame.copyTo(processor->frame);
                processor->detections = detections;
                processor->tracks = tracks;
                processor->has_new_frame = true;
            }
            else {
                std::cerr << "Camera " << camera_idx << " read error!" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        cap.release();
    };

    std::thread left_thread(process_camera, left_idx, left_processor, left_detector, std::ref(left_tracker));
    std::thread right_thread(process_camera, right_idx, right_processor, right_detector, std::ref(right_tracker));

    cv::namedWindow("Stereo Tracking - [Q] to quit", cv::WINDOW_NORMAL);
    cv::resizeWindow("Stereo Tracking - [Q] to quit", 2560, 960);

    cv::Mat display_left, display_right;
    StereoMatcher stereo_matcher(640.0f);

    std::map<int, int> left_super_ids;
    std::map<int, int> right_super_ids;
    int next_super_id = 0;

    while(true) {
        bool new_left = false, new_right = false;

        if(left_processor->has_new_frame && right_processor->has_new_frame) {
            std::lock_guard<std::mutex> lock_left(left_processor->mutex);
            std::lock_guard<std::mutex> lock_right(right_processor->mutex);

            auto stereo_pairs = stereo_matcher.matchTracks(
                left_processor->tracks,
                right_processor->tracks
            );

            for (const auto& pair : stereo_pairs) {
                int super_id;
                if (left_super_ids.count(pair.left_id)) {
                    super_id = left_super_ids[pair.left_id];
                } else if (right_super_ids.count(pair.right_id)) {
                    super_id = right_super_ids[pair.right_id];
                } else {
                    super_id = next_super_id++;
                }

                left_super_ids[pair.left_id] = super_id;
                right_super_ids[pair.right_id] = super_id;
            }

            left_processor->frame.copyTo(display_left);
            visualize_detections_and_tracks(display_left,
                                         left_processor->detections,
                                         left_processor->tracks,
                                         classes,
                                         &left_super_ids);
            left_processor->has_new_frame = false;
            new_left = true;

            right_processor->frame.copyTo(display_right);
            visualize_detections_and_tracks(display_right,
                                          right_processor->detections,
                                          right_processor->tracks,
                                          classes,
                                          &right_super_ids);
            right_processor->has_new_frame = false;
            new_right = true;

            if(!display_left.empty() && !display_right.empty()) {
                cv::Mat combined;
                cv::hconcat(display_left, display_right, combined);
                cv::imshow("Stereo Tracking - [Q] to quit", combined);
            }
        }

        if(cv::waitKey(1) == 'q') break;
    }

    left_processor->stop = true;
    right_processor->stop = true;

    left_thread.join();
    right_thread.join();

    cv::destroyAllWindows();
    return 0;
}