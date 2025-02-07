#include <iostream>
#include <opencv2/core/utils/logger.hpp>
#define RUN_PROGRAM2

#ifdef RUN_PROGRAM1
#include "StaticImage.h"
#endif

#ifdef RUN_PROGRAM2
#include "LiveCamera.h"
#endif

int main() {
cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
#ifdef RUN_PROGRAM1
    staticImageProto();
#endif
#ifdef RUN_PROGRAM2
    liveCameraProto();
#endif
    return 0;
}