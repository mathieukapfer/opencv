PWD=$(shell pwd)

POCL_CACHE_DIR=$(PWD)/.cache/pocl

MSG_CHECK_ENV_TOOCHAIN= $(if $(KALRAY_TOOLCHAIN_DIR), \
	"$(KALRAY_TOOLCHAIN_DIR)", \
	"is not set, please source ../kEnv/kvxtools/.switch_env")

MSG_CHECK_ENV_OPENCV_DATA= $(if $(wildcard $(OPENCV_TEST_DATA_PATH)*), \
	"$(OPENCV_TEST_DATA_PATH)", \
	"is not set")

help:
	@echo "======================="
	@echo "   Welcome to helper   "
	@echo "======================="
	@echo
	@echo "It will help you to quick install, configure, compile and run opencv test suite"
	@echo
	@echo " Initial setup: "
	@echo
	@echo "    Install required dependencies on your distribution:"
	@echo
	@echo "      make install-dep"
	@echo
	@echo "    Check your environment: "
	@echo
	@echo "      KALRAY_TOOLCHAIN_DIR:  $(MSG_CHECK_ENV_TOOCHAIN)"
	@echo "      OPENCV_TEST_DATA_PATH: $(MSG_CHECK_ENV_OPENCV_DATA)"
	@echo
	@echo " Configure and build the opencv lib with: "
	@echo
	@echo "      make configure            : create build dir and generate makefile"
	@echo "      make compile              : compile the opencl lib"
	@echo
	@echo " Then, build and run "
	@echo
	@echo "    Functional and Perf test: "
	@echo
	@echo "      make test_list            : list of binaries "
	@echo "      make run_gpu_<binary>     : execute test on gpu"
	@echo "      make run_mppa_<binary>    : execute test on mppa with - default conf "
	@echo
	@echo "      NOTE: <binary> : is the name of binary without 'opencv-' prefix"
	@echo "            <binary> : 'all_test' to execute the full test suite"
	@echo "            <binary> : 'all_perf' to execute the full perf suite"
	@echo
	@echo "    Special target"
	@echo "      make canny_perf_mppa      : execute canny test bench on mppa"
	@echo


# install dependencies
install-dep:
	sudo apt install libavcodec-dev libavformat-dev libavresample-dev libavutil-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgstreamermm-1.0-dev libgtk2.0-dev emacs

# clean
clean:
	rm -rf build


clean-pocl: clean-pocl-cache clean-pocl-cache-opencv

clean-pocl-cache:
	rm -rf $(POCL_CACHE_DIR)

clean-pocl-cache-opencv:
	rm -rf $(HOME)/.cache/opencv

# configure
opencv_prefix=install
opencv_test_bin_path=install

# NOTES:
#
#  - to enable opencl , we must have this
# 		OPENCL_FOUND                     ON
# 		WITH_OPENCL                      ON
#  - to build & run exampme, we must have this
# 		BUILD_EXAMPLES                   ON
# 		BUILD_PERF_TESTS                 ON
# 		INSTALL_BIN_EXAMPLES             ON
#  - to enable opencv trace, we can have
# 		CMAKE_BUILD_TYPE                 Debug
# 		OPENCV_TRACE                     ON
# 		OPENCV_TRACE_SYNC_OPENCL         ON
# 		ENABLE_INSTRUMENTATION           ON

configure:
	mkdir -p build
	cd build && cmake --debug-find ..  \
				-DCMAKE_EXPORT_COMPILE_COMMANDS=ON  \
				-DCMAKE_BUILD_TYPE=Release \
				-DCMAKE_TOOLCHAIN_FILE=./platforms/kvx/kvx.toolchain.cmake \
				-DCMAKE_C_FLAGS="$(shell pkg-config --cflags kaf-core)" \
				-DCMAKE_CXX_FLAGS="$(shell pkg-config --cflags kaf-core)" \
				-DCMAKE_INSTALL_PREFIX=${opencv_prefix}\
				-DOPENCV_TEST_INSTALL_PATH=${opencv_test_bin_path}\

configure-gpu:
	mkdir -p build
	cd build && cmake --debug-find ..  \
				-DCMAKE_EXPORT_COMPILE_COMMANDS=ON  \
				-DCMAKE_BUILD_TYPE=Debug \
				-DCMAKE_INSTALL_PREFIX=${opencv_prefix}\
				-DOPENCV_TEST_INSTALL_PATH=${opencv_test_bin_path}\


# trigger 'configure' the first time target 'compile' is called
build:
	make configure

# build
compile: build
	cd build &&	make -j`nproc`

# check mppa detection
clinfo:
	LD_PRELOAD=${KALRAY_TOOLCHAIN_DIR}/lib/libOpenCL.so	 clinfo

# ======================================================================================
# run environment
# ======================================================================================

COMMUN_ENV=	\
  PATH=$(PATH):$(PWD)/build/bin \
  LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(PWD)/build/lib \
	OPENCV_TEST_DATA_PATH=$(OPENCV_TEST_DATA_PATH) \
	POCL_CACHE_DIR=$(POCL_CACHE_DIR) \

ENABLE_MPPA_DEFAULT_MODE=	\
	OPENCV_OPENCL_DEVICE=':ACCELERATOR:'

ENABLE_GPU=	\
	OPENCV_OPENCL_DEVICE=':GPU:'

ENABLE_MPPA_IN_SPMD_MODE= \
	OPENCV_OPENCL_DEVICE=':ACCELERATOR:' \
	OPENCV_OPENCL_DEVICE_MAX_WORK_GROUP_SIZE=16

# ======================================================================================
# targets to list available test a perf
# ======================================================================================

.PHONY:test_list

test_list:
	ls build/bin | egrep "opencv_test|opencv_perf"

# ======================================================================================
# targets to run all *test* suite
# ======================================================================================

run_gpu_all_test:
	ls build/bin/opencv_test_* > test_list
	make -k $(foreach bin,  $(wildcard build/bin/opencv_test_*), $(bin:build/bin/opencv_%=run_gpu_%) ) | tee $@__.log
	grep "Run build/bin/\|PASSED\|FAILED\|=========="  $@__.log

run_mppa_all_test:
	ls build/bin/opencv_test_* > test_list
	make -k $(foreach bin,  $(wildcard build/bin/opencv_test_*), $(bin:build/bin/opencv_%=run_mppa_%)) | tee $@__.log
	grep "Run build/bin/\|PASSED\|FAILED\|=========="  $@__.log

# ======================================================================================
# targets to run all *perf* suite
# ======================================================================================

FORCE_SAMPLE:=1

run_gpu_all_perf:
	ls build/bin/opencv_perf_* > perf_list
	make -k $(foreach bin,  $(wildcard build/bin/opencv_perf_*), $(bin:build/bin/opencv_%=run_gpu_%) ) | tee $@__.log
	grep "Run build/bin/\|PASSED\|FAILED\|PERFSTAT\|RUN\|=========="  $@__.log
	grep "Run build/bin/\|PASSED\|FAILED\|=========="  $@__.log

run_mppa_all_perf:
	ls build/bin/opencv_perf_* > perf_list
	make -k	$(foreach bin,  $(wildcard build/bin/opencv_perf_*), $(bin:build/bin/opencv_%=run_mppa_%) ) | tee $@__.log
	grep "Run build/bin/\|PASSED\|FAILED\|PERFSTAT\|RUN\|=========="  $@__.log
	grep "Run build/bin/\|PASSED\|FAILED\|=========="  $@__.log

# ======================================================================================
# targets to parse test result
# ======================================================================================

check_failed:
	cat run_mppa_all_perf__.log run_mppa_all_test__.log  | grep  "Run build/bin/\|FAILED"

check_pbwgsize_%:
	grep "INVALID_WORK_GROUP_SIZE" run_mppa_all_$*__.log  | sed "s/.*clEnqueueNDRangeKernel(\(.*\))/\1/" | sed "s/globalsize=.*,//" | sort -u

checl_pbwgsize_all:
	cat  run_mppa_all_perf__.log run_mppa_all_test__.log  | grep "INVALID_WORK_GROUP_SIZE"| sed "s/.*clEnqueueNDRangeKernel(\(.*\))/\1/" | sed "s/globalsize=.*,//" | sort -u

# ======================================================================================
# targets to run only one test | perf
# ======================================================================================

# filter to unsupported test
run_mppa_perf_imgproc:build/bin/opencv_perf_imgproc
	${COMMUN_ENV} \
	${ENABLE_MPPA_IN_SPMD_MODE} \
	POCL_DEBUG=$(POCL_DEBUG) \
	$< --perf_force_samples=$(FORCE_SAMPLE) \
	--gtest_filter="-OCL_SqrBoxFilterFixture_SqrBoxFilter.SqrBoxFilter*"

# generic target for gpu
run_gpu_%: build/bin/opencv_%
	@echo
	@echo " Run $< as GPU device"
	@echo
	${COMMUN_ENV} \
	${ENABLE_GPU} \
	$< --perf_force_samples=$(FORCE_SAMPLE) \
	--gtest_filter="*CL*"

# generic target for mppa in default configuration
run_mppa_%: build/bin/opencv_%
	@echo
	@echo " Run $< as MPPA device in SPMD configuration"
	@echo
	${COMMUN_ENV} \
	${ENABLE_MPPA_IN_SPMD_MODE} \
	POCL_DEBUG=$(POCL_DEBUG) \
	$< --perf_force_samples=$(FORCE_SAMPLE) \
	--gtest_filter="*CL*"

# ======================================================================================
# Specific target for canny
# ======================================================================================

# enable POCL trace
# options
#POCL_DEBUG=all
#POCL_DEBUG=1
#POCL_DEBUG=0
POCL_DEBUG=timing

#GDB_ENABLE=gdb --args
#VALGRING=valgrind
#TIME=time

build_perf_canny:
		cd build && make opencv_perf_imgproc

parser-with-timing:
	emacs --batch --load $(PWD)/platforms/kvx/format_report.el --eval "(my-format-perf-report \"$(OUTPUT).log\")" | tee $(OUTPUT)_parsed.log
	@echo
	@echo "================================================================================="
	@echo
	@echo "Only with sobel:"
	cat $(OUTPUT)_parsed.log | grep stage1_with_sobel | column -t

canny_perf_gpu: build_perf_canny
	@echo
	@echo
	POCL_DEBUG=$(POCL_DEBUG) \
	OPENCV_OPENCL_DEVICE_MAX_WORK_GROUP_SIZE=256 \
	${COMMUN_ENV} \
  ${ENABLE_GPU} \
	${TIME} ${VALGRING} opencv_perf_imgproc --perf_force_samples=$(FORCE_SAMPLE) \
	--gtest_filter="*CL_CannyFixture_Canny.Canny*"  2>&1 | tee $@.log
	OUTPUT=$@ make parser-with-timing

canny_perf_mppa: build_perf_canny
	@echo
	@echo
	OPENCV_OPENCL_DEVICE_MAX_WORK_GROUP_SIZE=16 \
	POCL_DEBUG=$(POCL_DEBUG) \
	${COMMUN_ENV} \
  ${ENABLE_MPPA_IN_SPMD_MODE} \
	${TIME} ${VALGRING} opencv_perf_imgproc --perf_force_samples=$(FORCE_SAMPLE) \
	--gtest_filter="*CL_CannyFixture_Canny.Canny/*" 2>&1 | tee $@.log
	OUTPUT=$@ make parser-with-timing
