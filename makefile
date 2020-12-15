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
	cd build && LD_PRELOAD=${KALRAY_TOOLCHAIN_DIR}/lib/libOpenCL.so cmake --debug-find .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON  -DCMAKE_BUILD_TYPE=Release \
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


# ======================================================================================

COMMUN_ENV=	\
  PATH=$(PATH):$(PWD)/build/bin \
  LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(PWD)/build/lib \
	OPENCV_TEST_DATA_PATH=$(OPENCV_TEST_DATA_PATH) \

ENABLE_MPPA=	\
	OPENCV_OPENCL_DEVICE=':ACCELERATOR:'

ENABLE_GPU=	\
	OPENCV_OPENCL_DEVICE=':GPU:'

ENABLE_LW_MODE= \
	POCL_MPPA_FIRMWARE_NAME=ocl_fw_l2_d_1m_hugestack.elf \
	POCL_MPPA_EXTRA_EXEC_MODE=LW \
	POCL_MPPA_EXTRA_MAX_WORKGROUP_SIZE=256 \

ENABLE_SPMD_MODE= \
	OPENCV_OPENCL_DEVICE_MAX_WORK_GROUP_SIZE=16 \

# ======================================================================================
## test 'opencv_perf_photo'
### huge stack (& work groupe size = 256)
test_perf_photo-mppa-lw:
	${COMMUN_ENV} \
	${ENABLE_MPPA} \
  ${ENABLE_LW_MODE} \
	opencv_perf_photo --perf_force_samples=1

### work groupe size = 16
test_perf_photo-mppa:
	${COMMUN_ENV} \
	${ENABLE_MPPA} \
	opencv_perf_photo --perf_force_samples=1

# ======================================================================================
## perf_video
### LW mode
test_perf_video-mppa-lw:
	${COMMUN_ENV} \
  ${ENABLE_MPPA} \
  ${ENABLE_LW_MODE} \
	opencv_perf_video --perf_force_samples=1

### SMPD mode & work groupe size = 16
test_perf_video-mppa-spmd:
	${COMMUN_ENV} \
  ${ENABLE_MPPA} \
	${ENABLE_SPMD_MODE} \
	opencv_perf_video --perf_force_samples=1

### non mppa
test_perf_video-gpu:
	${COMMUN_ENV} \
	${ENABLE_GPU} \
	opencv_perf_video --perf_force_samples=1

# ======================================================================================
## test 'opencl-opencv-interop'
test-photo-lw:build/bin/opencv_test_photo
	${ENABLE_LW_MODE} \
	${COMMUN_ENV} \
	opencv_test_photo

test-photo:build/bin/opencv_test_photo
	${COMMUN_ENV} \
	opencv_test_photo


# ======================================================================================
## test video
test-video:build/bin/opencv_test_video
	${COMMUN_ENV} \
	${ENABLE_MPPA} \
	${ENABLE_SPMD_MODE} \
	opencv_test_video

# ======================================================================================
## test video
test-calib:build/bin/opencv_test_video
	${COMMUN_ENV} \
  opencv_test_calib3d

# ======================================================================================
## test 'opencl-opencv-interop'
build/bin/example_opencl_opencl-opencv-interop:
	cd build/samples/opencl && make

### enable POCL trace
POCL_DEBUG=1
#POCL_DEBUG=0

### work groupe size = 256
test-opencl-buffer:build/bin/example_opencl_opencl-opencv-interop
	cd build/bin &&  POCL_DEBUG=${POCL_DEBUG}	OPENCV_OPENCL_DEVICE=':ACCELERATOR:' ./example_opencl_opencl-opencv-interop --video=$(OPENCV_TEST_DATA_PATH)/cv/video/768x576.avi

### work groupe size = 16
test-opencl-buffer-16:build/bin/example_opencl_opencl-opencv-interop
	cd build/bin &&  POCL_DEBUG=${POCL_DEBUG} OPENCV_OPENCL_DEVICE_MAX_WORK_GROUP_SIZE=16	OPENCV_OPENCL_DEVICE=':ACCELERATOR:' ./example_opencl_opencl-opencv-interop --video=$(OPENCV_TEST_DATA_PATH)/cv/video/768x576.avi

#kvx-jtag-runner --reset
