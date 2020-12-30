PWD=$(shell pwd)
OpenCV_DIR=$(PWD)/../build
OpenCV_SRC=$(PWD)/..

# helper
# ----------
help:
	@echo
	@echo " configure: create build dir for compilation"
	@echo " compile:   compile all targets"
	@echo " run_<prg>: execute program"
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
POCL_DEBUG=1
#POCL_DEBUG=0

RUN_ENV_MPPA= \
	OPENCV_OPENCL_DEVICE=':ACCELERATOR:' \
	OPENCV_OPENCL_DEVICE_MAX_WORK_GROUP_SIZE=16 \
	POCL_DEBUG=${POCL_DEBUG} \

RUN_ENV_GPU= \
	OPENCV_OPENCL_DEVICE_MAX_WORK_GROUP_SIZE=16 \

RUN_ENV=$(RUN_ENV_GPU)

## run target
## ----------

run_%: build/%
	cd build && make $*
	$(RUN_ENV) \
	./build/$*


#IMAGE=$(OpenCV_SRC)/samples/data/graf3.png
transparent_api: build/$@
	cd build && make $*
	$(RUN_ENV) \
	./build/$@ $(IMAGE)


opencl_custom_kernel: build/$@
	cd build && make $*
	$(RUN_ENV) \
	./build/$@
