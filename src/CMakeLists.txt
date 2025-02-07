cmake_minimum_required(VERSION 3.30)

# ensure VCPKG_ROOT environmental variable points to your VCPKG installation
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "Vcpkg toolchain file")
else()
    message(FATAL_ERROR "Please set VCPKG_ROOT environment variable")
endif()

project(detection)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_ROOT "$ENV{VCPKG_ROOT}/installed/x64-windows/share/opencv4")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

file(GLOB OC_SORT_SOURCES "${CMAKE_SOURCE_DIR}/../../oc-sort/deploy/OCSort/cpp/src/*.cpp")

add_executable(detection
        main.cpp
        OneCamera.cpp
        YoloDetector.cpp
        YoloDetector.h
        OneCamera.h
        OCSortTracker.cpp
        OCSortTracker.h
        ${OC_SORT_SOURCES}
        StereoCamera.cpp
        StereoCamera.h
        StereoMatcher.h
        StereoMatcher.cpp
        hungarian.cpp
)

# include opencv include + libs
target_include_directories(detection PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(detection PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)

# include the assets folder in build
add_custom_command(TARGET detection POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:detection>/assets"
)