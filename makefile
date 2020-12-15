help:
	@echo "This is just an helper to quick install, configure, compile and run somes opencv examples"
	@echo
	@echo "First, follow this link for initial setup"
	@echo " https://github.com/mathieukapfer/howto/blob/master/howto_install_opencv.md"
	@echo
	@echo "Second, finalize the dependancies installation with"
	@echo "  make check-install"
	@echo
	@echo "Then, build & run the sample 'opencl-opencv-interop' with"
	@echo "  make test-opencl-buffer"

PWD=$(shell pwd)
OPENCV_TEST_DATA_PATH=$(PWD)/../opencv_extra/testdata
OPENCV_BIN_PATH=build/bin

COMMON_ENV= \
	OPENCV_TEST_DATA_PATH=$(OPENCV_TEST_DATA_PATH) \
	PATH=$(OPENCV_BIN_PATH)


# sudo apt list --installed
check-install:
	sudo apt install libavcodec-dev libavdevice58 libavformat-dev libavresample-dev libavutil-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgstreamermm-1.0-dev

build-test-opencl:
	cd build/samples/opencl && make

test-opencl-buffer:build-test-opencl
	$(COMMON_ENV) \
	example_opencl_opencl-opencv-interop --video=$(OPENCV_TEST_DATA_PATH)/cv/video/768x576.avi

test-video:
	$(COMMON_ENV) \
	opencv_test_video

test-photo:
	$(COMMON_ENV) \
	opencv_test_photo
