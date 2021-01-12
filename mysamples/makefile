PWD=$(shell pwd)
OpenCV_DIR=$(PWD)/../build
OpenCV_SRC=$(PWD)/..
#OpenCV_Extra_Data=$(PWD)/../../opencv_extra/testdata/cv
OpenCV_Extra_Data="/work2/common/embedded/KAF/KAF_libraries/opencv/4.5.0/opencv_extra/testdata/cv"

# helper
# ----------
help:
	@echo
	@echo " configure: create build dir for compilation"
	@echo " compile:   build all targets"
	@echo " <prg>:     build & execute one program"
	@echo

# configure
# ----------
configure:
	mkdir -p build && cd build && OpenCV_DIR=$(OpenCV_DIR)  cmake ..

# build
# ----------
compile:
	cd build && make


# run
# ----------


## run env
## ----------

# debug level
#POCL_DEBUG=all
#POCL_DEBUG=1
#POCL_DEBUG=0
POCL_DEBUG=timing

OCLGRING=oclgrind

RUN_ENV_MPPA= \
	OPENCV_OPENCL_DEVICE=':ACCELERATOR:' \
	OPENCV_OPENCL_DEVICE_MAX_WORK_GROUP_SIZE=16 \
	POCL_DEBUG=${POCL_DEBUG} \

RUN_ENV_GPU= \
	OPENCV_OPENCL_DEVICE_MAX_WORK_GROUP_SIZE=16 \

RUN_ENV=$(RUN_ENV_MPPA)

## run target
## ----------

#IMAGE=$(OpenCV_SRC)/samples/data/graf3.png
#IMAGE=$(OpenCV_Extra_Data)/shared/fruits.png
#IMAGE=$(OpenCV_Extra_Data)/shared/baboon.png
#IMAGE=$(OpenCV_Extra_Data)/shared/1_itseez-0001438.png
#IMAGE=$(OpenCV_Extra_Data)/../gpu/stereobm/aloe-L.png


## specific target
transparent_api: opencv_imgproc build/$@
	cd build && make $*
	$(RUN_ENV) \
	./build/$@ $(IMAGE)

## generic target
%: build/%
	cd build && make $*
	$(RUN_ENV) \
	$(OCLGRING) \
	./build/$*

## dependencies
opencv_imgproc:
	cd ../build/modules/imgproc/ && make $@
