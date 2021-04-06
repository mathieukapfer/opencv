# cmake needs this line
cmake_minimum_required(VERSION 3.1)

# Define project name
project(opencv_example_project)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Lsp mode
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# Declare the executable target built from your sources
add_executable(transparent_api transparent_api.cpp)

# Link your application with OpenCV libraries
target_link_libraries(transparent_api PRIVATE ${OpenCV_LIBS})


# Declare the executable target built from your sources
add_executable(opencl_custom_kernel opencl_custom_kernel.cpp)

# Link your application with OpenCV libraries
target_link_libraries(opencl_custom_kernel PRIVATE ${OpenCV_LIBS})

# Declare the executable target built from your sources
add_executable(opencl_fast_kernel opencl_fast_kernel.cpp)

# Link your application with OpenCV libraries
target_link_libraries(opencl_fast_kernel PRIVATE ${OpenCV_LIBS})

# Add opencl include and lib
set(OpenCL_INCLUDE_DIR $ENV{KALRAY_TOOLCHAIN_DIR})
message("OpenCL_INCLUDE_DIR=${OpenCL_INCLUDE_DIR}")

# Declare the executable target built from your sources
add_executable(opencl-opencv-interop opencl-opencv-interop.cpp)

# Link your application with OpenCV libraries
target_link_libraries(opencl-opencv-interop PRIVATE ${OpenCV_LIBS} $ENV{KALRAY_TOOLCHAIN_DIR}/lib/libOpenCL.so)

#set(CMAKE_MODULE_PATH=${KALRAY_TOOLCHAIN_DIR})
target_include_directories(opencl-opencv-interop SYSTEM PUBLIC ${OpenCL_INCLUDE_DIR}/include)
