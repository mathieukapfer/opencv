/**
 *****************************************************************************
 *
 * Copyright 2021 Kalray
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************
 */

#include "opencv2/core.hpp"
#include "opencv2/core/cvdef.h"
#include "opencv2/core/utils/logger.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core/opencl/ocl_defs.hpp"
#include <opencv2/core/ocl.hpp>
#include <bits/stdint-intn.h>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <chrono>
#include <list>
#include <mutex>
#include <thread>
#include <unistd.h>

using namespace cv;

// Select target:
//  - MatType = Mat  to run on CPU
//  - MatType = UMat to run on MPPA
using MatType = UMat;

#define NB_FRAMES 50

void capture_thread(std::mutex &frames_mtx, std::list<MatType> &frames,
        VideoCapture &capture)
{
    Mat frame;
    std::vector<Mat> channels;
    MatType channel;
    bool empty = false;
    do {
        frames_mtx.lock();
        if (frames.size() > 100) {
            frames_mtx.unlock();
            // Avoid interfering too much with the other threads' performance
            usleep(100);
            continue;
        }
        frames_mtx.unlock();

        capture >> frame;
        empty = frame.empty();
        if (empty) {
            capture.set(CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        // keep only 1 channel to have a gray frame
        cv::split(frame, channels);

        // put several frames to feed processing in order to avoid the capture
        // to be the bottleneck
        for (int i = 0; i < NB_FRAMES; i++) {
            channels[0].copyTo(channel);
            frames_mtx.lock();
            frames.push_back(std::move(channel));
            frames_mtx.unlock();
        }
    } while (true);
}

VideoCapture create_capture(std::string video)
{
    VideoCapture capture(video);
    if (!capture.isOpened()) {
        throw std::runtime_error("Invalid file " + video);
    }
    return capture;
}

enum class algorithm {
    CANNY,
    FAST,
    GAUSS,
};

struct fast_data {
    std::vector<cv::KeyPoint> keypoints;
};

struct frame_data {
    MatType in;
    MatType out;
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> end;
    algorithm algo;
    struct fast_data fast;
};


void display_thread(std::list<frame_data> &frames, std::mutex &frames_mtx)
{
    frame_data cur_frame;
    std::list<std::chrono::time_point<std::chrono::steady_clock>> frame_start_times;

    int x = 0;

    while (true) {
        frames_mtx.lock();
        if (frames.size() == 0) {
            frames_mtx.unlock();
            // Avoid interfering too much with the other threads' performance
            usleep(100);
            continue;
        }

        // Keep the timing of the 500 last frames. Only keep the last frame
        // since we may be displaying them slower than the rate at which we
        // get them.
        do {
            cur_frame = std::move(frames.front());
            frames.pop_front();
            frame_start_times.push_back(cur_frame.start);
            if (frame_start_times.size() > 500) {
                frame_start_times.pop_front();
            }
        } while (frames.size() != 0);
        frames_mtx.unlock();

        int nb_us = std::chrono::duration_cast<std::chrono::microseconds>(cur_frame.end - cur_frame.start).count();
        int total_nb_us = std::chrono::duration_cast<std::chrono::microseconds>(cur_frame.end - frame_start_times.front()).count() / frame_start_times.size();

        x++;
        // reduce host processing to be fair to the CPU implementation
        if ((x % 10) == 0) {
            cv::rectangle(cur_frame.out, {0, 0}, {400, 150}, 0, cv::FILLED);
            cv::putText(cur_frame.out,
                        "Average time: " + std::to_string(total_nb_us / 1000.0) +
                        "ms",
                        cv::Point(15, 50), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0,
                        cv::Scalar(255, 255, 255));
            cv::putText(cur_frame.out,
                        "FPS: " + std::to_string(1000000.0 / total_nb_us),
                        cv::Point(15, 100), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0,
                        cv::Scalar(255, 255, 255));

            Mat frame_to_display;
            cur_frame.out.copyTo(frame_to_display);
            if (cur_frame.algo == algorithm::FAST) {
                cv::drawKeypoints(frame_to_display, cur_frame.fast.keypoints,
                                  frame_to_display);
            }
            imshow("Output", frame_to_display);
            waitKey(1);
        }
    }
}

void parse_arg(int argc, const char *argv[], std::string &video, algorithm &algo) {

    if(argc < 3) {
        std::cout << std::endl << "  Usage: " << argv[0] << " [<video>] [<algo>] " << std::endl << std::endl;
        std::cout << std::endl << "  Available algo: CANNY, FAST, GAUSS " << std::endl << std::endl;
    } else {
        video = std::string(argv[1]);
        std::string input = std::string(argv[2]);
        if (input == "CANNY") {
            algo = algorithm::CANNY;
        } else if (input == "FAST") {
            algo = algorithm::FAST;
        } else if (input == "GAUSS") {
            algo = algorithm::GAUSS;
        } else {
            throw std::runtime_error("Invalid algorithm: " + input);
        }
    }
}

int main(int argc, const char *argv[]) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);

    // Get the path of the video to process and the algorithm to apply.
    std::string video;
    algorithm algo;
    parse_arg(argc, argv, video, algo);

    // Get a buffer pool.
    if (cv::ocl::useOpenCL()) {
        cv::ocl::getOpenCLAllocator()->getBufferPoolController()->setMaxReservedSize(512*1024*1024);
      }

    // Get images to process.
    std::list<MatType> in_frames;
    std::mutex in_frames_mtx;
    VideoCapture capture = create_capture(video);
    std::thread th([&in_frames_mtx, &in_frames, &capture](){
            capture_thread(in_frames_mtx, in_frames, capture);
            });

    // Display the processed images.
    std::list<frame_data> out_frames;
    std::mutex out_frames_mtx;
    std::thread display_th([&out_frames_mtx, &out_frames](){
            display_thread(out_frames, out_frames_mtx);
            });

    // Setup the module if needed.
    MatType img_in, img_out;
    frame_data data_frame;
    std::chrono::time_point<std::chrono::steady_clock> start_processing;
    std::chrono::time_point<std::chrono::steady_clock> stop_processing;

    // FAST specific instantiation.
    cv::Ptr<cv::Feature2D> detector = cv::FastFeatureDetector::create(20, false, FastFeatureDetector::TYPE_9_16);
    std::vector<cv::KeyPoint> keypoints;
    fast_data fast_frame;

    while (true) {

        start_processing = std::chrono::steady_clock::now();

        // Get a frame from the input ring.
        bool empty = true;
        while (empty) {
            in_frames_mtx.lock();
            empty = in_frames.size() == 0;
            if (!empty) {
                img_in = std::move(in_frames.front());
                in_frames.pop_front();
            }
            in_frames_mtx.unlock();
            if (empty) {
                usleep(100);
            }
        }

        if (img_in.empty()) {
            break;
        }

        // Process the frame using Kalray optimized OpenCL kernels.
        switch (algo) {
        case algorithm::CANNY:
            Canny(img_in, img_out, 50, 100);
            break;
        case algorithm::FAST:
            detector->detect(img_in, keypoints);
            img_out = img_in;
            break;
        case algorithm::GAUSS:
            GaussianBlur(img_in, img_out, Size(3, 3), 1.5, 1.5, BORDER_REPLICATE);
            break;
        default:
            throw std::runtime_error("Invalid");
        }

        stop_processing = std::chrono::steady_clock::now();

        // Add a frame to the output ring.
        fast_frame = {
            keypoints
        };

        data_frame = {
            std::move(img_in), std::move(img_out),
            start_processing, stop_processing,
            algo, fast_frame
        };

        out_frames_mtx.lock();
        out_frames.push_back(std::move(data_frame));
        out_frames_mtx.unlock();
    }
    return 0;
}
