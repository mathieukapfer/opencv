//source https://www.learnopencv.com/opencv-transparent-api/

#include "opencv2/core.hpp"
#include "opencv2/core/utils/logger.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <ostream>
using namespace cv;

#define DEFAULT_IMAGE "lena.jpg"
#define USE_TRANSPARENT_API

int main(int argc, char *argv[]) {
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_DEBUG);

  const char default_image_file[] = DEFAULT_IMAGE;
  const char * image = default_image_file;

  if(argc < 2) {
    std::cout << std::endl << "  Usage: " << argv[0] << " <image>" << std::endl << std::endl;
    exit(-1);
  } else {
    image = argv[1];
  }

#ifndef USE_TRANSPARENT_API
  Mat img, gray;
  img = imread(image, cv::IMREAD_COLOR);
#else
  std::cout << "Use UMAT => trigger transparent api mode" << std::endl;
  UMat img, gray;
  String filename(image);
  imread(filename, cv::IMREAD_COLOR).copyTo(img);
#endif
  // - time measurment init
  double t = (double)cv::getTickCount();

  cvtColor(img, gray, COLOR_BGR2GRAY);
  GaussianBlur(gray, gray, Size(7, 7), 1.5);
  Canny(gray, gray, 0, 50);

  // - time measurment
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  std::cout << "Times passed in seconds: " << t << std::endl;

  imshow("edges", gray);
  waitKey();

  return 0;
}
