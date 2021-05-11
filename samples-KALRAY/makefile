PWD=$(shell pwd)

OpenCV_DIR=$(PWD)/../build
OPENCV_SRC_DIR=$(PWD)/..

POCL_CACHE_DIR=$(PWD)/.cache/pocl
OPENCV_OPENCL_CACHE_DIR=$(PWD)/.cache


ifeq ($(KALRAY_TOOLCHAIN_DIR),)
	MSG_CHECK_ENV_TOOCHAIN:="not set"
else
	MSG_CHECK_ENV_TOOCHAIN:=$(KALRAY_TOOLCHAIN_DIR)
endif

ifeq ($(OPENCV_TEST_DATA_PATH),)
	MSG_CHECK_ENV_OPENCV_DATA:= \
	"should be set to \<path_to\>/opencv_extra/testdata \(https://github.com/opencv/opencv_extra.git\)"
else
	MSG_CHECK_ENV_OPENCV_DATA:=$(OPENCV_TEST_DATA_PATH)
endif

# helper
# ----------
help:
	@echo
	@echo " ========================================================================="
	@echo "  Welcome to this makefile helper that script the compilation & execution"
	@echo "  > If you edit the related source code locally or in ../modules/.. "
	@echo "  > just type 'make <prg>' to compile dependencies and execute it ;0) "
	@echo " ========================================================================="
	@echo
	@echo " Check your environment"
	@echo " ----------------------"
	@echo "   KALRAY_TOOLCHAIN_DIR:  $(MSG_CHECK_ENV_TOOCHAIN)"
	@echo "   OPENCV_TEST_DATA_PATH: $(MSG_CHECK_ENV_OPENCV_DATA)"
	@echo
	@echo " Selected input"
	@echo " ---------------"
	@echo "   IMAGE: $(IMAGE)"
	@echo "   VIDEO: $(VIDEO)"
	@echo
	@echo " Compile & execute"
	@echo " -----------------"
	@echo "   make configure:   create build dir for compilation (do it once)"
	@echo "   make compile:     build all targets (optional)"
	@echo "   make <prg>:       re-build dependecies & execute one program"
	@echo
	@echo "  The <prg> can be one of the following:"
	@echo "   - video_demo_*:           a demo of a video being processed by openCL"
	@echo "   - transparent_api:        a demo code of tapi that use BGR2GRAY, gaussian, canny "
	@echo "   - opencl_custom_kernel:   a demo to work on custom kernel (from opencv)"
	@echo "   - opencl_fast_kernel:     a demo to work on fast kernel (from opencv)"
	@echo "   - opencl-opencv-interop:  a demo to work on both opencl & opencv (from opencv)"
	@echo

# configure
# ----------
configure:
	mkdir -p build && cd build && OpenCV_DIR=$(OpenCV_DIR) \
	cmake .. \
				-DCMAKE_C_FLAGS="$(shell pkg-config --cflags kaf-core)" \
				-DCMAKE_CXX_FLAGS="$(shell pkg-config --cflags kaf-core)" \


# build
# ----------

## build all programs
compile:
	cd build && make

## dependencies
opencv_imgproc:
	cd ../build/modules/imgproc/ && make $@

opencv_features2d:
	cd ../build/modules/features2d && make $@

# parser
parser-with-timing:
	emacs --batch --load ${OPENCV_SRC_DIR}/platforms/kvx/format_report.el --eval "(my-format-perf-report \"$(OUTPUT).log\")" | tee $(OUTPUT)_parsed.log


# clean
# ----------
clean-pocl: clean-pocl-cache clean-pocl-cache-opencv

clean-pocl-cache:
	rm -rf $(POCL_CACHE_DIR)

clean-pocl-cache-opencv:
	rm -rf $(OPENCV_OPENCL_CACHE_DIR)

clean:
	rm -rf build

# run
# ----------


## run env
## ----------

# debug level
#POCL_DEBUG=all
#POCL_DEBUG=1
#POCL_DEBUG=0
#POCL_DEBUG=timing

RUN_COMMON= \
  OPENCV_OPENCL_CACHE_DIR=$(OPENCV_OPENCL_CACHE_DIR)

RUN_ENV_MPPA= \
	${RUN_COMMON} \
	OPENCV_OPENCL_DEVICE=':ACCELERATOR:' \
	OPENCV_OPENCL_DEVICE_MAX_WORK_GROUP_SIZE=16 \
	POCL_CACHE_DIR=$(POCL_CACHE_DIR) \
	POCL_DEBUG=${POCL_DEBUG}

RUN_ENV_GPU= \
	${RUN_COMMON} \
	OPENCV_OPENCL_DEVICE_MAX_WORK_GROUP_SIZE=16 \

RUN_ENV=$(RUN_ENV_MPPA)

## image & video as input
## ----------------------

#IMAGE=$(OPENCV_SRC_DIR)/samples/data/lena.jpg
#IMAGE=$(OPENCV_SRC_DIR)/samples/data/graf3.png
#IMAGE=$(OPENCV_TEST_DATA_PATH)/cv/shared/fruits.png
#IMAGE=$(OPENCV_TEST_DATA_PATH)/cv/shared/lena.png
#IMAGE=$(OPENCV_TEST_DATA_PATH)/cv/shared/baboon.png
#IMAGE=$(OPENCV_TEST_DATA_PATH)/cv/shared/1_itseez-0001438.png
IMAGE=$(OPENCV_TEST_DATA_PATH)/cv/../gpu/stereobm/aloe-L.png

#VIDEO=$(OPENCV_TEST_DATA_PATH)/highgui/video/big_buck_bunny.mp4
VIDEO=output.mp4

## create uncompressed HD video needed by video_demo from web ressources
output.mp4:
	wget https://test-videos.co.uk/vids/jellyfish/mp4/h264/1080/Jellyfish_1080_10s_1MB.mp4
	ffmpeg -i Jellyfish_1080_10s_1MB.mp4 -vcodec rawvideo -pix_fmt gray output.mp4

## run target
## ----------

## optional debug
#LAUNCH_GDB=gdb --args
#LAUNCH_GDB=valgrind --tool=memcheck

#default algorithm
VIDEO_DEMO_ALGORITHM ?=CANNY

video_demo: build/$@ opencv_imgproc opencv_features2d output.mp4
	cd build && make video_demo
	$(RUN_ENV) \
	./build/$@ $(VIDEO) $(VIDEO_DEMO_ALGORITHM)

video_demo_canny:
	VIDEO_DEMO_ALGORITHM=CANNY make video_demo

video_demo_gaussian:
	OPENCV_MPPA_MICROKERNEL_LIB=../platforms/kvx/build/libopencv_kalray_ukernels.a VIDEO_DEMO_ALGORITHM=GAUSS make video_demo

video_demo_fast:
	VIDEO_DEMO_ALGORITHM=FAST make video_demo

transparent_api: opencv_imgproc build/$@
	cd build && make $@
	$(RUN_ENV) \
	./build/$@ $(IMAGE) 2>&1 | tee $@.log
	OUTPUT=$@ make parser-with-timing

opencl_custom_kernel:
	cd build && make $@
	$(RUN_ENV) \
	./build/$@ 2>&1 | tee $@.log
	OUTPUT=$@ make parser-with-timing

opencl_fast_kernel:
	cd build && make $@
	$(RUN_ENV) \
	./build/$@ 2>&1 | tee $@.log
	OUTPUT=$@ make parser-with-timing

opencl-opencv-interop:
	cd build && make $@
	$(RUN_ENV) \
	./build/$@ 2>&1  --video=$(OPENCV_TEST_DATA_PATH)/cv/video/768x576.avi  | tee $@.log

## generic target
%: build/%
	cd build && make $@
	$(RUN_ENV) \
	./build/$*
