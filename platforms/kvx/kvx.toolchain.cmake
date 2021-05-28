# Sanity: disable Fortran anyway
SET(CMAKE_Fortran_COMPILER  "")

# Check Kalray toolchain
if(NOT DEFINED ENV{KALRAY_TOOLCHAIN_DIR})
	message(FATAL_ERROR "KALRAY_TOOLCHAIN_DIR env var is not set")
	return()
endif()

# What to build & install
SET(BUILD_EXAMPLES            ON  CACHE BOOL "Build all examples")
SET(BUILD_TESTS               ON  CACHE BOOL "Build accuracy & regression tests")
SET(BUILD_PERF_TESTS          ON  CACHE BOOL "Build performance tests")
SET(INSTALL_TESTS             ON  CACHE BOOL "Install accuracy and performance test binaries and test data")

# Disable AMD optimized libraries
SET(WITH_OPENCLAMDBLAS        OFF CACHE BOOL "Include AMD OpenCL BLAS library support")
SET(WITH_OPENCLAMDFFT         OFF CACHE BOOL "Include AMD OpenCL FFT library support")

# Disable Eigen backend
SET(WITH_EIGEN                OFF CACHE BOOL "Include Eigen2/Eigen3 support")

# Kalray OpenCL toolchain
SET(WITH_OPENCL               ON  CACHE BOOL "Include OpenCL Runtime support")
SET(WITH_OPENCL_SVM           OFF CACHE BOOL "Include OpenCL Shared Virtual Memory support")
SET(OPENCL_INCLUDE_DIR        "$ENV{KALRAY_TOOLCHAIN_DIR}/include"          CACHE PATH     "Custom vendor OpenCL header path")
SET(OPENCL_LIBRARY            "$ENV{KALRAY_TOOLCHAIN_DIR}/lib/libOpenCL.so" CACHE FILEPATH "Custom vendor OpenCL library")

# Generate .pc
SET(OPENCV_GENERATE_PKGCONFIG ON  CACHE BOOL "Generate .pc file for pkg-config build tool")

# Not supported modules
SET(BUILD_opencv_dnn          OFF CACHE BOOL "Build DNN module")
SET(BUILD_opencv_ml           OFF CACHE BOOL "Build ML module")

SET(MPPA_lib_src "${CMAKE_CURRENT_LIST_DIR}/mppa_conv_3x3.c")
SET(MPPA_lib_obj "${CMAKE_CURRENT_LIST_DIR}/build/libopencv_kalray_ukernels.o")
SET(MPPA_lib_out "${CMAKE_CURRENT_LIST_DIR}/build/libopencv_kalray_ukernels.a")

if(NOT KVX_TARGETS_DECLARED)
    SET(KVX_TARGETS_DECLARED TRUE)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/build/")

    add_custom_command(OUTPUT "${MPPA_lib_obj}" MAIN_DEPENDENCY "${MPPA_lib_src}"
                       COMMAND "$ENV{KALRAY_TOOLCHAIN_DIR}/bin/clang" -O2 -funroll-loops -fPIC -c -o "${MPPA_lib_obj}" "${MPPA_lib_src}")

    add_custom_command(OUTPUT "${MPPA_lib_out}" MAIN_DEPENDENCY "${MPPA_lib_obj}"
                       COMMAND "$ENV{KALRAY_TOOLCHAIN_DIR}/bin/kvx-cos-ar" rcs "${MPPA_lib_out}" "${MPPA_lib_obj}")

    add_custom_target(MPPA_build_lib ALL DEPENDS "${MPPA_lib_out}")

    install(FILES "${MPPA_lib_out}" DESTINATION "kvx-cos/lib/")
endif(NOT KVX_TARGETS_DECLARED)
