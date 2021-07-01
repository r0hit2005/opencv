#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <assert.h>

using namespace std;
using namespace cv;

//! [convolution-sequential]
void conv_seq(Mat src, Mat &dst, Mat kernel)
{
    int rows = src.rows, cols = src.cols;

    dst = Mat(rows, cols, src.type());

    // Taking care of edge values
    // Make border = ksize / 2;

    int n = kernel.rows, sz = n / 2;
    copyMakeBorder(src, src, sz, sz, sz, sz, BORDER_REPLICATE);

    for (int i = 0; i < rows; i++)
    {
        uchar* dptr = dst.ptr(i);

        for (int j = 0; j < cols; j++)
        {
            float value = 0;
            for (int k = -sz; k <= sz; k++)
            {
                uchar* sptr = src.ptr(i + sz + k);
                for (int l = -sz; l <= sz; l++)
                {
                    value += kernel.at<float>(k + sz, l + sz)*sptr[j + sz + l];
                }
            }
            dptr[j] = value;
        }
    }
}
//! [convolution-sequential]


#ifdef CV_CXX11
//! [convolution-parallel-cxx11]
void conv_parallel(Mat src, Mat &dst, int depth, Mat kernel)
{
    int rows = src.rows, cols = src.cols;

    dst = Mat(rows, cols, CV_8UC1, Scalar(0));

    // Taking care of edge values
    // Make border = ksize / 2;

    int ksize = kernel.rows, sz = ksize / 2;
    copyMakeBorder(src, src, sz, sz, sz, sz, BORDER_REPLICATE);


    parallel_for_(Range(0, rows*cols), [&](const Range& range)
    {
        for (int r = range.start; r < range.end; r++)
        {
            int i = r / cols, j = r % cols;

            double value = 0;
            for (int k = -sz; k <= sz; k++)
            {
                for (int l = -sz; l <= sz; l++)
                {
                    value += kernel.ptr(k + sz)[l + sz]*src.ptr(i + sz + k)[j + sz + l];
                }
            }
            dst.ptr(i)[j] = saturate_cast<uchar>(value);
        }

    });
}
//! [convolution-parallel-cxx11]
#else
//! [convolution-parallel]
class parallelConvolution : public ParallelLoopBody
{
private:
    
    Mat m_src, &m_dst;
    Mat m_kernel;

public:

    parallelConvolution(Mat src, Mat &dst, Mat kernel)
        : m_src(src), m_dst(dst), m_kernel(kernel)
    {} 

    virtual void operator()(const Range &range) const CV_OVERRIDE
    {
        for (int r = range.start; r < range.end; r++)
        {
            int i = r / m_src.cols, j = r % m_src.cols;

            double value = 0;
            for (int k = -(m_kernel.rows/2); k <= m_kernel.rows / 2; k++)
            {
                for (int l = -m_kernel.rows/2; l <= m_kernel.rows/2; l++)
                {
                    value += m_kernel.ptr(k + m_kernel.rows/2)[l + m_kernel.rows/2]*m_src.ptr(i + m_kernel.rows/2 + k)[j + m_kernel.rows/2 + l];
                }
            }
            m_dst.ptr(i)[j] = saturate_cast<uchar>(value);
        }
    }
};
//! [convolution-parallel]

//! [convolution-parallel-function]
void conv_parallel(Mat src, Mat &dst, Mat kernel)
{
    int rows = src.rows, cols = src.cols;

    dst = Mat(rows, cols, CV_8UC1, Scalar(0));

    // Taking care of edge values
    // Make border = ksize / 2;

    int ksize = kernel.rows, sz = ksize / 2;
    copyMakeBorder(src, src, sz, sz, sz, sz, BORDER_REPLICATE);

    parallelConvolution obj(src, dst, kernel);
    parallel_for_(Range(0, rows*cols), obj);
}
//! [convolution-parallel-function]
#endif

int main()
{
    Mat src = imread("lena.jpg", 0), dst1, dst2, kernel;
    imshow("Input", src);

    kernel = (Mat_<float>(5, 5) <<  1, 1, 1, 1, 1, 
                                    1, 1, 1, 1, 1, 
                                    1, 1, 1, 1, 1, 
                                    1, 1, 1, 1, 1, 
                                    1, 1, 1, 1, 1);
    kernel /= 25;

    cout << "Sequential implementation: ";
    conv_seq(src, dst1, kernel);
    imshow("Output1", dst1);
    waitKey(0);

    cout << "Parallel implementation: ";
    conv_parallel(src, dst1, kernel);
    imshow("Output1", dst1);
    waitKey(0);
    

    return 0;
}