PWD=$(shell pwd)
OpenCV_DIR=$(PWD)/../build
OpenCV_SRC=$(PWD)/..

# helper
help:
	@echo
	@echo " configure: create build dir for compilation"
	@echo " compile:   compile all targets"
	@echo " run_<prg>: execute program"
	@echo

# configure
configure:
	mkdir -p build && cd build && OpenCV_DIR=$(OpenCV_DIR)  cmake ..

# build
compile:
	cd build && make


# run

# uncomment pocl debug level
#POCL_DEBUG=all
POCL_DEBUG=1
#POCL_DEBUG=0

RUN_ENV= \
	OPENCV_OPENCL_DEVICE=':ACCELERATOR:' \
	POCL_DEBUG=${POCL_DEBUG} \

#OPENCV_OPENCL_DEVICE_MAX_WORK_GROUP_SIZE=16 \

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
