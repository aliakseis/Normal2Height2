// Normal2Height2.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


#include <lbfgs.h>

#include <iostream>

namespace {

double sqr(double x) { return x * x; };


int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
)
{
    std::cout << "Iteration: " << k << "; step: " << step << "; fx: " << fx << "; xnorm: " << xnorm << "; gnorm: " << gnorm << '\n';
    return 0;
}

lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *x_,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
)
{
    auto& normals_ = *static_cast<cv::Mat*>(instance);

    auto const width = normals_.cols;
    auto const height = normals_.rows;

    int residual_idx = 0;

    memset(g, 0, sizeof(lbfgsfloatval_t) * width * height);

    double fx = 0;

    for (int y = 1; y < height - 1; ++y)
    {
        for (int x = 1; x < width - 1; ++x)
        {
            auto kx = (x_[y * width + x + 1] - x_[y * width + x - 1]) / 2;
            auto ky = (x_[(y + 1) * width + x] - x_[(y - 1) * width + x]) / 2;

            auto out = normals_.at<cv::Vec3f>(y, x);

            auto kx_wanted = -out[0] / out[2];
            auto ky_wanted = -out[1] / out[2];

            auto dkx = kx_wanted - kx;
            auto dky = ky_wanted - ky;

            fx += dkx * dkx + dky * dky;

            g[y * width + x + 1] -= dkx;
            g[y * width + x - 1] += dkx;

            g[(y + 1) * width + x] -= dky;
            g[(y - 1) * width + x] += dky;
        }
    }

    return fx;
};

}


int main(int argc, char* argv[])
{
    std::string in_file =
        argv[1];
    if (in_file.empty())
    {
        std::cout << "Couldn't locate " << in_file << std::endl;
        return 1;
    }

    auto in_tex = cv::imread(in_file);

    auto const width = in_tex.cols;
    auto const height = in_tex.rows;

    /*
    std::vector<cv::Mat> bgr;
    split(in_tex, bgr);

    cv::imshow("x", bgr[2]);
    cv::imshow("y", bgr[1]);
    cv::imshow("z", bgr[0]);

    cv::waitKey();
    */

    // generate clean normal map

    cv::Mat normals(in_tex.size(), CV_32FC3);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            auto p = in_tex.at<cv::Vec3b>(y, x);
            auto& out = normals.at<cv::Vec3f>(y, x);
            if (p[0] == 255 && p[1] == 255 && p[2] == 255)
            {
                out[0] = 0;
                out[1] = 0;
                out[2] = 1;
            }
            else
            {
                auto nx = p[2] - 127.5;
                auto ny = -(p[1] - 127.5);
                auto nz = p[0] - 127.5;

                auto coeff = 1. / sqrt(sqr(nx) + sqr(ny) + sqr(nz));
                out[0] = nx * coeff;
                out[1] = ny * coeff;
                out[2] = nz * coeff;
            }
        }
    }

    const auto n_samples = normals.rows * normals.cols;


    // Initialize solution vector
    lbfgsfloatval_t fx;
    lbfgsfloatval_t *x = lbfgs_malloc(n_samples);
    if (x == nullptr) {
        return EXIT_FAILURE;
    }
    for (int i = 0; i < n_samples; i++) {
        x[i] = 0;
    }

    // Initialize the parameters for the optimization.
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    //param.orthantwise_c = param_c; // this tells lbfgs to do OWL-QN
    //param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    int lbfgs_ret = lbfgs(n_samples, x, &fx, evaluate, progress, &normals, &param);

    cv::Mat Xat2(normals.rows, normals.cols, CV_64FC1, x);

    cv::Mat copy;

    normalize(Xat2, copy, 0, 1, cv::NORM_MINMAX);

    lbfgs_free(x);

    cv::imshow("result", copy);

    cv::waitKey();

    return 0;
}
