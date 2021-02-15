PWD=$(shell pwd)
OpenCV_DIR=$(PWD)/../build
OPENCV_SRC_DIR=$(PWD)/..

POCL_CACHE_DIR=$(PWD)/.cache/pocl

MSG_CHECK_ENV_TOOCHAIN= $(if $(KALRAY_TOOLCHAIN_DIR), \
	"$(KALRAY_TOOLCHAIN_DIR)", \
	"is not set, please source ../../kEnv/kvxtools/.switch_env")

MSG_CHECK_ENV_OPENCV_DATA= $(if $(wildcard $(OPENCV_TEST_DATA_PATH)*), \
	"$(OPENCV_TEST_DATA_PATH)", \
	"is not set")

# helper
# ----------
help:
	@echo
	@echo " make configure:   create build dir for compilation"
	@echo " make compile:     build all targets"
	@echo " make <prg>:       build & execute one program"
	@echo
	@echo " <prg> "
	@echo "  transparent_api:        a demo code of tapi that use BRD2GARY, gaussian, canny "
	@echo "  opencl_custom_kernel:   a demo to work on custom kernel (from opencv)"
	@echo "  opencl-opencv-interop:  a demo to work on both opencl & opencv (from opencv)"
	@echo
	@echo "    Check your environment: "
	@echo
	@echo "      KALRAY_TOOLCHAIN_DIR:  $(MSG_CHECK_ENV_TOOCHAIN)"
	@echo "      OPENCV_TEST_DATA_PATH: $(MSG_CHECK_ENV_OPENCV_DATA)"
	@echo

# configure
# ----------
configure:
	mkdir -p build && cd build && OpenCV_DIR=$(OpenCV_DIR)  cmake ..

# build
# ----------
compile:
	cd build && make

# cleann
# ----------
clean-pocl: clean-pocl-cache clean-pocl-cache-opencv

clean-pocl-cache:
	rm -rf $(POCL_CACHE_DIR)

clean-pocl-cache-opencv:
	rm -rf $(HOME)/.cache/opencv

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

RUN_ENV_MPPA= \
	OPENCV_OPENCL_DEVICE=':ACCELERATOR:' \
	OPENCV_OPENCL_DEVICE_MAX_WORK_GROUP_SIZE=16 \
	POCL_CACHE_DIR=$(POCL_CACHE_DIR) \
	POCL_DEBUG=${POCL_DEBUG} \

RUN_ENV_GPU= \
	OPENCV_OPENCL_DEVICE_MAX_WORK_GROUP_SIZE=16 \

RUN_ENV=$(RUN_ENV_MPPA)

## run target
## ----------



## specific target

#IMAGE=$(OPENCV_SRC_DIR)/samples/data/lena.jpg
#IMAGE=$(OPENCV_SRC_DIR)/samples/data/graf3.png
#IMAGE=$(OPENCV_TEST_DATA_PATH)/cv/shared/fruits.png
#IMAGE=$(OPENCV_TEST_DATA_PATH)/cv/shared/baboon.png
#IMAGE=$(OPENCV_TEST_DATA_PATH)/cv/shared/1_itseez-0001438.png
#IMAGE=$(OPENCV_TEST_DATA_PATH)/cv/../gpu/stereobm/aloe-L.png

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

opencl-opencv-interop:
	cd build && make $@
	$(RUN_ENV) \
	./build/$@ 2>&1  --video=$(OPENCV_TEST_DATA_PATH)/cv/video/768x576.avi  | tee $@.log
	OUTPUT=$@ make parser-with-timing

## generic target
%: build/%
	cd build && make $@
	$(RUN_ENV) \
	./build/$*

## dependencies
opencv_imgproc:
	cd ../build/modules/imgproc/ && make $@

# parser
parser-with-timing:
	emacs --batch --load ${OPENCV_SRC_DIR}/platforms/kvx/format_report.el --eval "(my-format-perf-report \"$(OUTPUT).log\")" | tee $(OUTPUT)_parsed.log
