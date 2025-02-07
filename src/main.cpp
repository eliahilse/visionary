#include <iostream>
#include <opencv2/core/utils/logger.hpp>
#include "../../oc-sort/deploy/OCSort/cpp/include/OCSort.hpp"
#define RUN_PROGRAM2

#ifdef RUN_PROGRAM2
#include "StereoCamera.h"
#endif


int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
#ifdef RUN_PROGRAM2
    stereoCameraProto();
#endif
    return 0;
}