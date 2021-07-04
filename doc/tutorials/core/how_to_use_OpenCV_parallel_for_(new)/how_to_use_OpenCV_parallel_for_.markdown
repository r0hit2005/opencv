How to use the OpenCV parallel_for_ to parallelize your code {#tutorial_how_to_use_OpenCV_parallel_for_}
==================================================================

@tableofcontents

@prev_tutorial{tutorial_file_input_output_with_xml_yml}
@next_tutorial{tutorial_univ_intrin}

|    |    |
| -: | :- |
| Compatibility | OpenCV >= 3.0 |

Goal
----

The goal of this tutorial is to show you how to use the OpenCV `parallel_for_` framework to easily
parallelize your code. To illustrate the concept, we will write a program to perform a convolution operation over an image.
The full tutorial code is [here]().


Precondition
----

### Parallel Frameworks
The first precondition is to have OpenCV built with a parallel framework.
In OpenCV 4.5, the following parallel frameworks are available in that order:

*   Intel Threading Building Blocks (3rdparty library, should be explicitly enabled)
*   C= Parallel C/C++ Programming Language Extension (3rdparty library, should be explicitly enabled)
*   OpenMP (integrated to compiler, should be explicitly enabled)
*   APPLE GCD (system wide, used automatically (APPLE only))
*   Windows RT concurrency (system wide, used automatically (Windows RT only))
*   Windows concurrency (part of runtime, used automatically (Windows only - MSVC++ >= 10))
*   Pthreads

As you can see, several parallel frameworks can be used in the OpenCV library. Some parallel libraries(e.g. TBB, C=) are third party libraries and have to be explicitly enabled in CMake before building, while others are automatically available with the platform (e.g. APPLE GCD). 


### Race Conditions
Race conditions occur when more than one thread try to write *or* read and write to a particular memory location simultaneously. 
Based on that, we can broadly classify algorithms into two categories:- 
1. Algorithms in which only a single thread writes data to a particular memory location.  
    * In *convolution*, for example, even though multiple  threads may read from a pixel at a particular time, only a single thread *writes* to a particular pixel.
    <!-- * Some other example. -->
<!-- <br> -->

2. Algorithms in which multiple threads may write to a single memory location. 
    * Finding contours, features, etc. Such algorithms may require each thread to add data to a global variable simultaneously. For example, when detecting features, each thread will add features of their respective parts of the image to a common vector, thus creating a race condition.
    * We'll demonstrate a simple case in this tutorial. 


Convolution 
-----------

We will use the example of performing a convolution to demonstrate the use of parallel_for_ to parallelize the computation. This is an example of an algorithm which does not lead to a race condition.

Theory
------
Convolution is a simple mathematical operation widely used in image processing. Here, we slide a smaller matrix, called the *kernel*, over an image and a sum of the product of pixel values and corresponding values in the kernel gives us the value of the particular pixel in the output (called the anchor point of the kernel).  Based on the values in the kernel, we get different results. 
In the example below, we use a 3x3 kernel (anchored at its center) and convolve over a 5x5 matrix to produce a 3x3 matrix. The size of the output can be altered by padding the input with suitable values.
![Convolution Animation](images/convolution-example-matrix.gif)
For more information about different kernels and what they do, look [here](https://docs.opencv.org/4.5.2/d7/da8/tutorial_table_of_content_imgproc.html).

For the purpose of this tutorial, we will implement the simplest form of the function which takes a grayscale image (1 channel) and an odd length square kernel and produces an output image. 
The operation will not be performed in-place. 
@note In order to perform in-place, the only feasible method is to create a temporary matrix, store the values, and then copy the values to the original image. 

For an *n-sized kernel*, we will add a border of size *n/2* to handle edge cases.

Pseudocode
-----------

```
InputImage src, OutputImage dst, kernel(size n)
makeborder(src, n/2)
for each pixel (i, j) strictly inside borders, do:
{
    sum := 0
    for k := -n/2 to n/2, do:
        for l := -n/2 to n/2, do:
            sum += kernel[n/2 + k][n/2 + l]*src[i + k][j + l]
        
    dst[i][j] := sum
}
```

Implementation
--------------

Sequential implementation
--------------------------

@snippet how_to_use_OpenCV_parallel_for_.cpp convolution-sequential

In this implementation, we first make a dst image with the same size as the src image and add borders to the src image.
@snippet how_to_use_OpenCV_parallel_for_.cpp convolution-make-borders

We then sequentially iterate over the pixels in the src image and compute the value over the kernel and the neighbouring pixel values. 
@snippet how_to_use_OpenCV_parallel_for_.cpp convolution-kernel-loop

We then fill value to the corresponding pixel in the dst image.
@snippet how_to_use_OpenCV_parallel_for_.cpp fill



Parallel implementation
--------------------------

When looking at the sequential implementation, we can notice that each pixel depends on multiple neighbouring pixels but only one pixel is edited at a time. Thus, to optimize the computation, we can split the image into stripes and parallely perform convolution on each, by exploiting the multi-core architecture of modern processor. The OpenCV @ref cv::parallel_for_ framework automatically decides how to split the computation efficiently and does most of the work for us.

@note Although values of a pixel in a particular stripe may depend on pixel values outside the stripe, these are only read only operations and hence will not cause undefined behaviour.


The first thing is to declare a custom class that inherits from @ref cv::ParallelLoopBody and to override the
`virtual void operator ()(const cv::Range& range) const`.
@snippet how_to_use_OpenCV_parallel_for_.cpp convolution-parallel

The range in the `operator ()` represents the subset of values that will be treated by an individual thread. Based on the requirement, there may be different ways of splitting the range which in turn changes the computation.

For example, we can either 
1. Split the entire traversal of the image and obtain the [row, col] coordinate in the following way (as shown in the above code):
    ```
    for(int r = range.start; r < range.end; r++)
    {
        int i = r / cols, j = r % cols;
        // Similar to sequential implementation with the corresponding i and j values
    }
    ```
    @snippet how_to_use_OpenCV_parallel_for_.cpp overload-full

    We would then call the parallel_for_ function in the following way:
    @snippet how_to_use_OpenCV_parallel_for_.cpp convolution-parallel-function
<br>

2. Split the rows and compute for each row:
    ```
    In this case, the range is from [0, rows]. Each 
    for(int i = range.start ; i < range.end ; i++)
    {
        for(int j = 0 ; j < cols ; j++)
        {
            // Similar to sequential implementation with the corresponding i and j values
        }
    }
    ```
    @snippet how_to_use_OpenCV_parallel_for_.cpp overload-row-split

    In this case, we call the parallel_for_ function with a different range:
    @snippet how_to_use_OpenCV_parallel_for.cpp convolution-parallel-function-row


To set the number of threads, you can use: @ref cv::setNumThreads. You can also specify the number of splitting using the nstripes parameter in @ref cv::parallel_for_. For instance, if your processor has 4 threads, setting `cv::setNumThreads(2)` or setting `nstripes=2` should be the same as by default it will use all the processor threads available but will split the workload only on two threads.

@note C++ 11 standard allows to simplify the parallel implementation by get rid of the `ParallelMandelbrot` class and replacing it with lambda expression:

@snippet how_to_use_OpenCV_parallel_for_.cpp convolution-parallel-cxx11

Results
-----------



You can find the full tutorial code [here]().

The performance of the parallel implementation depends of the type of CPU you have. For instance, on 4 cores - 8 threads CPU, runtime may be 6x to 7x faster than a sequential implementation. There are many factors to explain why we do not achieve a speed-up of 8x:

*   the overhead to create and manage the threads,
*   background processes running in parallel,
*   the difference between 4 hardware cores with 2 logical threads for each core and 8 hardware cores.

In the tutorial, we used a normalizing filter which produces a blurred output.

<style>
.row{
    display:flex;
}
.col{
    width:50%;
    max-width: 500px;
    padding: 10px;
    text-align:center;
}
</style>

<div class="row">
    <div class="col">
        <img src="images/input.jpeg">
        Input
    </div>
    <div class="col">
        <img src="images/output.jpeg">
        Output
    </div>
</div>



