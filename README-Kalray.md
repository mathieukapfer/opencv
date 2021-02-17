
## How to use OpenCV with MPPA platform

 - Get the makefile helper thanks to dynamic link:

      $ ln -s platforms/kvx/makefile .

 - Display the help with:

       $ make

 - Check your environment variables

      KALRAY_TOOLCHAIN_DIR:   <path_to_kalray/accesscore>
      OPENCV_TEST_DATA_PATH:  <path_to_opencv_extra/testdata>


 - As explained in the help, to reproduce Canny bench and result parsing, do

       $ make configure
       $ make compile
       $ make canny_perf_mppa

 - Some other examples are also available in ./samples-KALRAY
       $ cd ./samples-KALRAY
       $ make configure
       $ make compile
       $ make opencl-opencv-interop
       $ make opencl_custom_kernel
       $ make transparent_api
