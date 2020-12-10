help:
	@echo "This is just an helper to quick install, configure, compile and run somes opencv examples"
	@echo
	@echo " Prerequis:"
	@echo
	@echo " - Check the your distribution with"
	@echo "      make check-install"
	@echo
	@echo " - To run on mppa, you need to source the kalray env"
	@echo "   Do once (or when you change KAF_LIBRARIES version:"
	@echo "      cd .. && ./get_packages.sh && cd -"
	@echo
	@echo "   Do on each session:"
	@echo "      source ../kEnv/kvxtools/.switch_env"
	@echo
	@echo
	@echo " Then, configure and build the opencv lib with: "
	@echo "      make configure            : create build dir and generate makefile"
	@echo "      make compile              : compile the opencl lib"
	@echo
	@echo " Then, build and run somes examples: "
	@echo "      make test-opencv_perf_photo-mppa : build & run the 'opencv_perf_photo'"
	@echo "      make test-opencl-buffer          : build & run the 'example_opencl_opencl-opencv-interop'"
	@echo

# user setting
PWM=$(shell pwd)
OPENCV_TEST_DATA_PATH= "/work2/common/embedded/KAF/KAF_libraries/opencv/4.5.0/opencv_extra/testdata"

# check dependencies
#  (sudo apt list --installed)
check-install:
	sudo apt install libavcodec-dev libavformat-dev libavresample-dev libavutil-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgstreamermm-1.0-dev libgtk2.0-dev

# clean
clean:
	rm -rf build;

#cd build && make clean

# configure
opencv_prefix=install
opencv_test_bin_path=install

# NOTE: to build & run example, we need this
#
# BUILD_EXAMPLES                   ON
# BUILD_PERF_TESTS                 ON
# INSTALL_BIN_EXAMPLES             ON
# OPENCL_FOUND                     ON
# WITH_OPENCL                      ON
configure:
	mkdir -p build
	cd build && LD_PRELOAD=${KALRAY_TOOLCHAIN_DIR}/lib/libOpenCL.so cmake --debug-find ..   -DCMAKE_BUILD_TYPE=Release \
                    -DBUILD_EXAMPLES=ON \
	                  -DBUILD_PERF_TESTS=ON \
                    -DCMAKE_C_FLAGS="$(shell pkg-config --cflags kaf-core)" \
                    -DCMAKE_CXX_FLAGS="$(shell pkg-config --cflags kaf-core)" \
                    -DWITH_OPENCL=ON \
                    -DHAVE_OPENCL_STATIC=OFF \
                    -DOPENCL_INCLUDE_DIR=$(KALRAY_TOOLCHAIN_DIR)/include \
                    -DOPENCL_LIBRARIES=$(KALRAY_TOOLCHAIN_DIR)/lib/libOpenCL.so \
                    -DOPENCV_GENERATE_PKGCONFIG=ON \
                    -DBUILD_TESTS=ON -DBUILD_PERF_TESTS=ON -DINSTALL_TESTS=ON\
                    -DCMAKE_INSTALL_PREFIX=${opencv_prefix}\
                    -DOPENCV_TEST_INSTALL_PATH=${opencv_test_bin_path}\
                    -DOpenCL_INCLUDE_DIR=$(KALRAY_TOOLCHAIN_DIR)/include \
                    -DOpenCL_LIBRARY=$(KALRAY_TOOLCHAIN_DIR)/lib/libOpenCL.so \


# trigger 'configure' the first time target 'compile' is called
build:
	make configure

# build
compile: build
	cd build && LD_PRELOAD=${KALRAY_TOOLCHAIN_DIR}/lib/libOpenCL.so	 make -k -j 4

# build and execute

## test 'opencv_perf_photo'
test-opencv_perf_photo-mppa:
	OPENCV_TEST_DATA_PATH=$(OPENCV_TEST_DATA_PATH) \
  PATH=$(PATH):$(PWD)/build/bin \
  LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(PWD)/build/lib \
	POCL_MPPA_FIRMWARE_NAME=ocl_fw_l2_d_1m_hugestack.elf \
	POCL_MPPA_EXTRA_EXEC_MODE=LW \
	POCL_MPPA_EXTRA_MAX_WORKGROUP_SIZE=256 \
	OPENCV_OPENCL_DEVICE=':ACCELERATOR:' \
	opencv_perf_photo


## test 'opencl-opencv-interop'
build/bin/example_opencl_opencl-opencv-interop:
	cd build/samples/opencl && make

test-opencl-buffer:build/bin/example_opencl_opencl-opencv-interop
	cd build/bin &&  OPENCV_OPENCL_DEVICE_MAX_WORK_GROUP_SIZE=16	OPENCV_OPENCL_DEVICE=':ACCELERATOR:' ./example_opencl_opencl-opencv-interop --video=$(OPENCV_TEST_DATA_PATH)/cv/video/768x576.avi

#kvx-jtag-runner --reset
