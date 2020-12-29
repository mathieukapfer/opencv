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
add_executable(opencv_example example.cpp)

# Link your application with OpenCV libraries
target_link_libraries(opencv_example PRIVATE ${OpenCV_LIBS})


# Declare the executable target built from your sources
add_executable(my_example my_example.cpp)

# Link your application with OpenCV libraries
target_link_libraries(my_example PRIVATE ${OpenCV_LIBS})


# Declare the executable target built from your sources
add_executable(my_example_filter my_example_filter.cpp)

# Link your application with OpenCV libraries
target_link_libraries(my_example_filter PRIVATE ${OpenCV_LIBS})


#/usr/bin/c++
set(CMAKE_CXX_FLAGS "-D_USE_MATH_DEFINES -D__OPENCV_BUILD=1 -D__OPENCV_TESTS=1 -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -I/home/kapfer/Project/opencv/build/3rdparty/ippicv/ippicv_lnx/icv/include -I/home/kapfer/Project/opencv/build/3rdparty/ippicv/ippicv_lnx/iw/include -I/home/kapfer/Project/opencv/build -I/home/kapfer/Project/opencv/modules/ts/include -I/home/kapfer/Project/opencv/modules/imgproc/include -I/home/kapfer/Project/opencv/modules/imgcodecs/include -I/home/kapfer/Project/opencv/modules/videoio/include -I/home/kapfer/Project/opencv/modules/core/include -I/home/kapfer/Project/opencv/modules/highgui/include  -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Winit-self -Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -fvisibility=hidden -fvisibility-inlines-hidden -O0 -g -DDEBUG   -fPIE   -std=c++11")

#set(CMAKE_CXX_FLAGS "-D_USE_MATH_DEFINES -D__OPENCV_BUILD=1 -D__OPENCV_TESTS=1 -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -I/home/kapfer/Project/opencv/build/3rdparty/ippicv/ippicv_lnx/icv/include -I/home/kapfer/Project/opencv/build/3rdparty/ippicv/ippicv_lnx/iw/include -I/home/kapfer/Project/opencv/build -I/home/kapfer/Project/opencv/modules/ts/include -I/home/kapfer/Project/opencv/modules/imgproc/include -I/home/kapfer/Project/opencv/modules/imgcodecs/include -I/home/kapfer/Project/opencv/modules/videoio/include -I/home/kapfer/Project/opencv/modules/core/include -I/home/kapfer/Project/opencv/modules/highgui/include  -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Winit-self -Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fvisibility-inlines-hidden -O3 -DNDEBUG  -DNDEBUG -fPIE   -std=c++11")

#set(CMAKE_CXX_FLAGS="-fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Winit-self -Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fvisibility-inlines-hidden -O3 -DNDEBUG  -DNDEBUG")

# Declare the executable target built from your sources
add_executable(mandelbrote mandelbrote.cpp)

# Link your application with OpenCV libraries
target_link_libraries(mandelbrote PRIVATE ${OpenCV_LIBS})


# Declare the executable target built from your sources
add_executable(transparent_api transparent_api.cpp)

# Link your application with OpenCV libraries
target_link_libraries(transparent_api PRIVATE ${OpenCV_LIBS})


# Declare the executable target built from your sources
add_executable(opencl_custom_kernel opencl_custom_kernel.cpp)

# Link your application with OpenCV libraries
target_link_libraries(opencl_custom_kernel PRIVATE ${OpenCV_LIBS})
