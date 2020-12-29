
//source https://www.learnopencv.com/opencv-transparent-api/

#include "opencv2/core.hpp"
#include "opencv2/core/cvdef.h"
#include "opencv2/core/utils/logger.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <bits/stdint-intn.h>
#include <iomanip>
#include <iostream>
#include <ostream>
using namespace cv;

#define DEFAULT_IMAGE "lena.jpg"
#define USE_TRANSPARENT_API

#define TIMESTAMP_START                                                 \
  static int64 tick_ref = (int64)cv::getTickCount();                    \
  std::cout << "Start" << std::endl;

#define TIMESTAMP                                                       \
  {                                                                     \
    int64 tick = (int64)cv::getTickCount();                             \
    double t = double(tick - tick_ref) / cv::getTickFrequency();        \
    std::cout << std::fixed << std::setprecision(6) << t << " ["        \
              << std::fixed << std::setprecision(6) << tick - tick_ref  \
              << "] " << __FUNCTION__ << ":" << __LINE__ << std::endl;   \
  }


int main(int argc, char *argv[]) {
  //cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_DEBUG);
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);

  const char default_image_file[] = DEFAULT_IMAGE;
  const char * image = default_image_file;

  if(argc < 2) {
    std::cout << std::endl << "  Usage: " << argv[0] << " [<image>] " << std::endl << std::endl;
  } else {
    image = argv[1];
  }

  TIMESTAMP_START;

#ifndef USE_TRANSPARENT_API
  Mat img, gray;
  img = imread(image, cv::IMREAD_COLOR);
#else
  std::cout << "Use UMAT => trigger transparent api mode" << std::endl;
  UMat img, gray;
  String filename(image);
  imread(filename, cv::IMREAD_COLOR).copyTo(img);
#endif
  TIMESTAMP;

  cvtColor(img, gray, COLOR_BGR2GRAY); TIMESTAMP;
  GaussianBlur(gray, gray, Size(7, 7), 1.5); TIMESTAMP;
  Canny(gray, gray, 0, 50); TIMESTAMP;

  imshow("edges", gray);
  waitKey();

  return 0;
}
