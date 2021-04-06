// source: samples/tapi/opencl_custom_kernel.cpp

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    const char* keys =
        "{ i input    | | specify input image }"
        "{ h help     | | print help message }";

    cv::CommandLineParser args(argc, argv, keys);
    if (args.has("help"))
    {
        cout << "Usage : " << argv[0] << " [options]" << endl;
        cout << "Available options:" << endl;
        args.printMessage();
        return EXIT_SUCCESS;
    }

    cv::ocl::Context ctx = cv::ocl::Context::getDefault();
    if (!ctx.ptr())
    {
        cerr << "OpenCL is not available" << endl;
        return 1;
    }
    cv::ocl::Device device = cv::ocl::Device::getDefault();
    if (!device.compilerAvailable())
    {
        cerr << "OpenCL compiler is not available" << endl;
        return 1;
    }


    UMat src;
    UMat result;
    {
        Mat frame(cv::Size(640, 480), CV_8U, Scalar::all(128));
        Point p(frame.cols / 2, frame.rows / 2);
        line(frame, Point(0, frame.rows / 2), Point(frame.cols, frame.rows / 2), 1);
        circle(frame, p, 200, Scalar(32, 32, 32), 8, LINE_AA);
        string str = "OpenCL";
        int baseLine = 0;
        Size box = getTextSize(str, FONT_HERSHEY_COMPLEX, 2, 5, &baseLine);
        putText(frame, str, Point((frame.cols - box.width) / 2, (frame.rows - box.height) / 2 + baseLine),
                FONT_HERSHEY_COMPLEX, 2, Scalar(255, 255, 255), 5, LINE_AA);
        frame.copyTo(src);
        frame.copyTo(result);
    }


    cv::String module_name; // empty to disable OpenCL cache

    {
        cv::Ptr<cv::Feature2D> detector = cv::FastFeatureDetector::create(10, false, FastFeatureDetector::TYPE_9_16);
        std::vector<cv::KeyPoint> keypoints;
        detector->detect(src, keypoints);

        if (keypoints.size() > 0) {
            cv::drawKeypoints(src, keypoints, result);
        }

        imshow("Source", src);
        imshow("Result", result);

        waitKey();
    }
    return 0;
}
