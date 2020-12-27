#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"

int main( )
{

  cv::Mat image(100, 100, CV_8UC3, cv::Scalar(128,0,0));
  cv::Mat sameImage = image;
  cv::Mat anotherImage = image.clone();

  // fill with random value
  cv::randu(anotherImage, 0, 255);

  // image display as text - on a smaller region of interest
  //std::cout << image;
  cv::Rect r(0, 0, 5, 5);
  std::cout << image(r); ;

  // image display as window
  cv::imshow("an image", image);
  cv::imshow("another image", anotherImage);
  cv::waitKey(0);

  // access each pixel in a 3 channel image
  // - time measurment init
  double t = (double)cv::getTickCount();

  // - treatment
  for (int i=0; i < image.rows; i++) {
    for (int j=0; j < image.cols; j++) {
      cv::Mat_<cv::Vec3b> _I = image;
      _I(i,j)[0] = (i+j)%256;
      _I(i,j)[1] = (i+j)%256;
      _I(i,j)[2] = (i+j)%256;
    }
  }

  // - time measurment
  t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
  std::cout << "Times passed in seconds: " << t << std::endl;

  cv::imshow("", image);
  cv::waitKey(0);

  return 0;
}
