cmake_minimum_required(VERSION 3.16)
project(DistanceBetweenObjects VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tell CMake where OpenCV is installed
set(OpenCV_DIR "C:/Users/Asus/source/repos/DistanceBetweenObjects/opencv/build/x64/vc16/lib")

find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Create the executable from your source files
add_executable(DistanceBetweenObjects
    DistanceBetweenObjects.cpp
 "SendDataToCloud.cpp")

# Link OpenCV to your program
target_link_libraries(DistanceBetweenObjects
    ${OpenCV_LIBS}
)
