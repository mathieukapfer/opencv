PWM=$(shell pwd)

# default user settings
#OPENCV_TEST_DATA_PATH?="${PWD}/../opencv_extra/testdata"
OPENCV_TEST_DATA_PATH?="/work2/common/embedded/KAF/KAF_libraries/opencv/4.5.0/opencv_extra/testdata/"

help:
	@echo "======================="
	@echo "   Welcome to helper   "
	@echo "======================="
	@echo
	@echo "It will help you to quick install, configure, compile and run some opencv examples"
	@echo
	@echo "Initial setup: "
	@echo " - Check your distribution:"
	@echo "      make check-install"
	@echo
	@echo "Initial setup [kalray dev mode only]:"
	@echo " - Source the kalray env"
	@echo "   Do once (or when you change KAF_LIBRARIES version)"
	@echo "      cd .. && ./get_packages.sh && cd -"
	@echo
	@echo "   Do on each session:"
	@echo "      source ../kEnv/kvxtools/.switch_env"
	@echo
	@echo
	@echo " Configure and build the opencv lib with: "
	@echo "      make configure            : create build dir and generate makefile"
	@echo "      make compile              : compile the opencl lib"
	@echo
	@echo " Then, build and run "
	@echo
	@echo "    Functional and Perf test: "
	@echo
	@echo "      make test_list            : list of binary "
	@echo "      make test_<binary>        : execute test on mppa with default conf "
	@echo "      make test_<binary>_lw     : execute test on mppa with LW conf [experimental only]"
	@echo
	@echo "    Some examples: "
	@echo "      make test-opencl-buffer   : build & run the 'example_opencl_opencl-opencv-interop'"
	@echo
	@echo " You can change the opencv data path with OPENCV_TEST_DATA_PATH env variable"
	@echo " Current value: $(OPENCV_TEST_DATA_PATH)"


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

#CMAKE_BUILD_TYPE=Debug
#OPENCV_TRACE
#OPENCV_TRACE_SYNC_OPENCL
#ENABLE_INSTRUMENTATION

configure:
	mkdir -p build
	cd build && LD_PRELOAD=${KALRAY_TOOLCHAIN_DIR}/lib/libOpenCL.so cmake --debug-find ..  \
										-DCMAKE_EXPORT_COMPILE_COMMANDS=ON  \
										-DCMAKE_BUILD_TYPE=Release \
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


# trigger 'configure' the first time target 'compile' is called
build:
	make configure

# build
compile: build
	cd build &&	make -k -j 4

#compile: build
#	cd build && LD_PRELOAD=${KALRAY_TOOLCHAIN_DIR}/lib/libOpenCL.so	 make -k -j 4

# check mppa detection
clinfo:
	LD_PRELOAD=${KALRAY_TOOLCHAIN_DIR}/lib/libOpenCL.so	 clinfo

# ======================================================================================
#POCL_MPPA_EXTRA_BUILD_CFLAGS='-O0 -g'

COMMUN_ENV=	\
  PATH=$(PATH):$(PWD)/build/bin \
  LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(PWD)/build/lib \
	OPENCV_TEST_DATA_PATH=$(OPENCV_TEST_DATA_PATH) \
	POCL_CACHE_DIR=$(PWM)/.cache/pocl \

ENABLE_MPPA_DEFAULT_MODE=	\
	OPENCV_OPENCL_DEVICE=':ACCELERATOR:'

ENABLE_GPU=	\
	OPENCV_OPENCL_DEVICE=':GPU:'

ENABLE_MPPA_IN_LW_MODE= \
	OPENCV_OPENCL_DEVICE=':ACCELERATOR:' \
	POCL_MPPA_FIRMWARE_NAME=ocl_fw_l2_d_1m_hugestack.elf \
	POCL_MPPA_EXTRA_EXEC_MODE=LW \
	POCL_MPPA_EXTRA_MAX_WORKGROUP_SIZE=256 \
  POCL_MPPA_EXTRA_BUILD_CFLAGS=$(POCL_MPPA_EXTRA_BUILD_CFLAGS) \

ENABLE_MPPA_IN_SPMD_MODE= \
	OPENCV_OPENCL_DEVICE=':ACCELERATOR:' \
	OPENCV_OPENCL_DEVICE_MAX_WORK_GROUP_SIZE=16 \
  POCL_MPPA_EXTRA_BUILD_CFLAGS=$(POCL_MPPA_EXTRA_BUILD_CFLAGS) \

# ======================================================================================
FORCE_SAMPLE:=1
#POCL_DEBUG=all
#POCL_DEBUG=1
#POCL_DEBUG=0

.PHONY:test_list

test_list:
	ls build/bin | egrep "opencv_test|opencv_perf"

test_mppa_all_test:
	ls build/bin/opencv_test_* > test_list
	make	$(foreach bin,  $(wildcard build/bin/opencv_test_*), $(bin:build/bin/opencv_%=test_mppa_%) )

test_gpu_%: build/bin/opencv_%
	@echo
	@echo " Run $< as GPU device"
	@echo
	${COMMUN_ENV} \
	${ENABLE_GPU} \
	$< --perf_force_samples=$(FORCE_SAMPLE) \
	--gtest_filter="*CL*"

test_mppa_%: build/bin/opencv_%
	@echo
	@echo " Run $< as MPPA device in SPMD configuration"
	@echo
	${COMMUN_ENV} \
	${ENABLE_MPPA_IN_SPMD_MODE} \
	POCL_DEBUG=$(POCL_DEBUG) \
	$< --perf_force_samples=$(FORCE_SAMPLE) \
	--gtest_filter="*CL*"

test_mppa_%_lw: build/bin/opencv_%
	@echo
	@echo " Run $< as MPPA device in LW configuration [Experimental ONLY]"
	@echo
	${COMMUN_ENV} \
  ${ENABLE_MPPA_IN_LW_MODE} \
	$< --perf_force_samples=$(FORCE_SAMPLE) \
	--gtest_filter="*CL*"

test_perf: build/bin/opencv_perf_photo
	@echo
	@echo " Run $< as MPPA device in SPMD configuration"
	@echo
	${COMMUN_ENV} \
	${ENABLE_MPPA_IN_SPMD_MODE} \
	POCL_DEBUG=1 \
	$< --perf_force_samples=$(FORCE_SAMPLE) \
	--gtest_filter="OCL_Photo_DenoisingColored.DenoisingColored"

test_perf_gpu: build/bin/opencv_perf_photo
	@echo
	@echo " Run $< as MPPA device in SPMD configuration"
	@echo
	${COMMUN_ENV} \
	${ENABLE_GPU} \
	POCL_DEBUG=1 \
	$< --perf_force_samples=$(FORCE_SAMPLE) \
	--gtest_filter="OCL_Photo_DenoisingColored.DenoisingColored"

build/bin/opencv_perf_%:
	cd build/modules/$* && make $(notdir $@)

build/bin/opencv_test_%:
	cd build/modules/$* && make $(notdir $@)

# ======================================================================================
# WIP

### enable POCL trace
# options


#GDB_ENABLE=gdb --args
#VALGRING=valgrind
#TIME=time

test_ok_:
	@echo
	@echo " SPECIAL CONF !!!! "
	@echo "Run $< in SPMD configuration"
	@echo
	${COMMUN_ENV} \
  ${ENABLE_MPPA_IN_LW_MODE} \
	${TIME} ${VALGRING} opencv_perf_core --perf_force_samples=$(FORCE_SAMPLE) \
	--gtest_filter="*OCL_*"

test_ok:
	@echo
	@echo " SPECIAL CONF !!!! "
	@echo
	${COMMUN_ENV} \
  ${ENABLE_MPPA_IN_LW_MODE} \
	POCL_DEBUG=${POCL_DEBUG} \
	${GDB_ENABLE}	\
	${VALGRING} opencv_perf_core --perf_force_samples=$(FORCE_SAMPLE) \
	--gtest_filter="-*OCL_BufferPoolFixture_BufferPool_UMatCanny*"

test_ko:
	@echo
	@echo " SPECIAL CONF !!!! "
	@echo
	${COMMUN_ENV} \
  ${ENABLE_MPPA_IN_SPMD_MODE} \
	POCL_DEBUG=${POCL_DEBUG} \
	${GDB_ENABLE} opencv_perf_core --perf_force_samples=$(FORCE_SAMPLE) \
	--gtest_filter="*OCL_BufferPoolFixture_BufferPool_UMatCanny10*"


# ======================================================================================


build/bin/example_opencl_opencl-opencv-interop:
	cd build/samples/opencl && make

### work groupe size = 256
test-opencl-buffer:build/bin/example_opencl_opencl-opencv-interop
	${COMMUN_ENV} \
	example_opencl_opencl-opencv-interop --video=$(OPENCV_TEST_DATA_PATH)/cv/video/768x576.avi

### work groupe size = 16
test-opencl-buffer-mppa-spmd:build/bin/example_opencl_opencl-opencv-interop
	${COMMUN_ENV} \
	${ENABLE_MPPA_IN_SPMD_MODE} \
	POCL_DEBUG=${POCL_DEBUG} example_opencl_opencl-opencv-interop --video=$(OPENCV_TEST_DATA_PATH)/cv/video/768x576.avi

#kvx-jtag-runner --reset
