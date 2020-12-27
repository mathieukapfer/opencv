// sample from:
// opencv / samples / cpp / tutorial_code / core /  mat_mask_operations
//
// compile & execute with:
// cd build && make && OPENCV_SAMPLES_DATA_PATH=~/Project/opencv/samples/data ./my_opencv_example_filter

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

static void help(char* progName)
{
    cout << endl
        <<  "This program shows how to filter images with mask: the write it yourself and the"
        << "filter2d way. " << endl
        <<  "Usage:"                                                                        << endl
        << progName << " [image_path -- default lena.jpg] [G -- grayscale] "        << endl << endl;
}

int main( int argc, char* argv[])
{
    help(argv[0]);
    const char* filename = argc >=2 ? argv[1] : "lena.jpg";

    Mat src, dst0, dst1;

    if (argc >= 3 && !strcmp("G", argv[2]))
        src = imread( samples::findFile( filename ), IMREAD_GRAYSCALE);
    else
        src = imread( samples::findFile( filename ), IMREAD_COLOR);

    if (src.empty())
    {
        cerr << "Can't open image ["  << filename << "]" << endl;
        return EXIT_FAILURE;
    }

    namedWindow("Input", WINDOW_AUTOSIZE);
    namedWindow("Output", WINDOW_AUTOSIZE);

    imshow( "Input", src );

    //![kern]
    Mat kernel = (Mat_<char>(3,3) <<  0, -1,  0,
                                   -1,  5, -1,
                                    0, -1,  0);
    //![kern]
    double t = (double)getTickCount();

    //![filter2D]
    filter2D( src, dst1, src.depth(), kernel );
    //![filter2D]
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Built-in filter2D time passed in seconds:     " << t << endl;

    imshow( "Output", dst1 );

    waitKey();
    return EXIT_SUCCESS;
}
