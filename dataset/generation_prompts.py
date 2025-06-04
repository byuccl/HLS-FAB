desired_test = "matrix_vector_multiply_float32"
desired_test_description = (
    """
    Multiplies a 32-bit floating-point matrix by a 32-bit floating-point vector, resulting in a 32-bit floating-point vector.
    """
)


desired_test = "accumulate_opencv"
desired_test_description = (
    """
The accumulate function adds an image (src1) to the accumulator image (src2), and generates the accumulated result image (dst).

dst[i] = |src1[i] + src2[i]|

dst[i] is the i-th element of the output array.
src1[i] is the i-th element of the first input array.
src2[i] is the i-th element of the second input array.


The input image will be 128x128 pixels
Each pixel of the output image is represented by a 1-channel, 8-bit, unsigned integer
Each pixel of the output image is represented by a 1-channel, 16-bit, unsigned integer
The image will be passed in and out as single-row arrays. Defined constants for 'HEIGHT' and 'WIDTH' (both set to 128) should be used to correctly break the array into a 2D image.
Implementation details should match the OpenCV Reduce implementation

void accumulate_opencv(uint8_t*)in_gray.data, (uint8_t*)in_gray1.data, (uint8_t*)dst_hls.data);
    """
)


desired_test = "reduce_avg_opencv"
desired_test_description = (
    """
The Reduce function reduces the matrix to a vector by treating rows/cols as set of 1-D vectors and performing specified operation on vectors until a single row/col is obtained.

Specific functionality for this kernel will be:
REDUCE_AVG : The output is the mean vector of all of the matrix’s rows/columns.

The input image will be 128x128 pixels
The 'dim' parameter represents the dimension index along which the matrix is reduced. 0 means that the matrix is reduced to a single row. 1 means that the matrix is reduced to a single column.
Each pixel of the image is represented by an 8-bit unsigned integer.
The image will be passed in and out as single-row arrays. Defined constants for 'HEIGHT' and 'WIDTH' (both set to 128) should be used to correctly break the array into a 2D image.
Implementation details should match the OpenCV Reduce implementation

void reduce_avg_opencv((uint8_t*)in_img.data, dimension, (uint8_t*)dst_hls.data);
    """
)


desired_test = "sobel_filter_opencv"
desired_test_description = (
    """
The sobel_filter_opencv function Computes the gradients of input image in both x and y direction by convolving the kernel with input image being processed.

The input image will be 128x128 pixels
The filter is a standard 3x3 Sobel filter
Each pixel of the image is represented by an 8-bit unsigned integer.
The images will be passed in and out as single-row arrays. Defined constants for 'ROWS' and 'COLS' (both set to 128) should be used to correctly break the array into a 2D image.
Implementation details should match the OpenCV Sobel Filter implementation
The border pixels should be set to a constant 0

void sobel_filter_opencv(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                         ap_uint<OUTPUT_PTR_WIDTH>* img_out1,
                         ap_uint<OUTPUT_PTR_WIDTH>* img_out2);
    """
)