// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "../perf_precomp.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
enum AcceleratorKind {
    CPU,
    ACCELERATOR,
};

CV_ENUM(AcceleratorEnabled, CPU, ACCELERATOR)
CV_ENUM(TestedBorderTypes, BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101)

typedef tuple<AcceleratorEnabled, Size, Size, TestedBorderTypes> OCL_bench_gauss_t;
typedef perf::TestBaseWithParam<OCL_bench_gauss_t> OCL_bench_gauss;

PERF_TEST_P(OCL_bench_gauss, gauss,
            testing::Combine(
                AcceleratorEnabled::all(),
                ::testing::Values(cv::Size(256, 256), szVGA, cv::Size(512, 512), sz720p, sz1080p, sz2160p, cv::Size(1920*4, 1080*4), cv::Size(1920*8, 1080*8)),
                ::testing::Values(cv::Size(3, 3)),
                TestedBorderTypes::all()
                )
            )
{
    cv::Size imgSize = get<1>(GetParam());
    cv::Size filterSize = get<2>(GetParam());

    Mat _img = imread(getDataPath("gpu/stereobm/aloe-L.png"), cv::IMREAD_GRAYSCALE);

    if (_img.empty())
        FAIL() << "Unable to load source image.";

    if (!get<0>(GetParam())) {
        Mat img;
        cv::resize(_img, img, imgSize, 0, 0, INTER_LINEAR_EXACT);
        Mat out(img.size(), img.type());

        declare.in(img).out(out);

        PERF_SAMPLE_BEGIN();
            GaussianBlur(img, out, filterSize, 1.5, 1.5, get<3>(GetParam()));
        PERF_SAMPLE_END();

        SANITY_CHECK(out);
    } else {
        UMat img;
        cv::resize(_img, img, imgSize, 0, 0, INTER_LINEAR_EXACT);
        UMat out(img.size(), img.type());

        declare.in(img).out(out);

        PERF_SAMPLE_BEGIN();
            GaussianBlur(img, out, filterSize, 1.5, 1.5, get<3>(GetParam()));
        PERF_SAMPLE_END();

        SANITY_CHECK(out);
    }

}

} // namespace

#endif // HAVE_OPENCL
