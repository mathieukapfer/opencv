PWD=$(shell pwd)
OpenCV_DIR=$(PWD)/../build
OpenCV_SRC=$(PWD)/..

help:
	@echo
	@echo " configure: create build dir for compilation"
	@echo " compile:   compile all targets"
	@echo " run_<prg>: execute program"
	@echo

configure:
	mkdir -p build && cd build && OpenCV_DIR=$(OpenCV_DIR)  cmake ..

compile:
	cd build && make

RUN_ENV= \

#OPENCV_OPENCL_DEVICE_MAX_WORK_GROUP_SIZE=16 \

run_%: build/%
	cd build && make $*
	$(RUN_ENV) \
	./build/$*

transparent_api: build/$@
	cd build && make $*
	$(RUN_ENV) \
	./build/$@ $(OpenCV_SRC)/samples/data/graf3.png
