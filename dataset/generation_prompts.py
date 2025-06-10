desired_test = "bilateral_filter_opencv"
desired_test_description = (
    """
In general, any smoothing filter smoothens the image which will affect the edges of the image. To preserve the edges while smoothing, a bilateral filter can be used. In an analogous way as the Gaussian filter, the bilateral filter also considers the neighboring pixels with weights assigned to each of them. These weights have two components, the first of which is the same weighing used by the Gaussian filter. The second component takes into account the difference in the intensity between the neighboring pixels and the evaluated one.

The bilateral filter applied on an image is:  
BF[I]_p = (1/W_p) ∑_(q∈S) G_σ_s(||p-q||)G_σ_r(||I_p-I_q||)I_q  
  
Where  
W_p = ∑_(q∈S) G_σ_s(||p-q||)G_σ_r(||I_p-I_q||)  
  
and G_σ is a Gaussian filter with variance σ.  
  
The Gaussian filter is given by:  
G_σ = e^(-(x^2+y^2)/(2σ^2))

The input image will be 128x128 pixels
Each pixel of the output image is represented by a 1-channel, 8-bit, unsigned integer
Each pixel of the output image is represented by a 1-channel, 8-bit, unsigned integer
The filter size is 3.
The image will be passed in and out as single-row arrays. Defined constants for 'HEIGHT' and 'WIDTH' (both set to 128) should be used to correctly break the array into a 2D image.
Implementation details should match the OpenCV implementation unless otherwise specified

void bilateral_filter_opencv((uint8_t*)in_img.data, sigma_color, sigma_space, (uint8_t*)out_img.data);
    """
)

desired_test = "bad_pixel_correction_opencv"
desired_test_description = (
"""
The Bad Pixel Correction function identifies and removes anomalous pixels in an image that significantly differ from their neighbors, often resulting from sensor defects, manufacturing faults, or temperature variations. It uses a specific cross-diagonal pattern to identify and correct these bad pixels.

The algorithm applies a 5x5 cross-diagonal mask with the following pattern:
1 0 1 0 1
0 0 0 0 0
1 0 0 0 1
0 0 0 0 0
1 0 1 0 1

This mask is used to detect pixels that are either too bright (hot pixels) or too dark (dead pixels) compared to their neighbors. If a pixel's value is lower than its surrounding minimum or higher than its surrounding maximum, it will be adjusted accordingly.

Key processing steps:

Create minimum filtered version (erode) of the input image using the cross-diagonal mask
Create maximum filtered version (dilate) of the input image using the same mask
Calculate the difference between the input and these filtered images
Apply thresholding to remove negative values
Correct the output by subtracting hot pixel differences and adding dead pixel differences
The input image will be 128x128 pixels
Each pixel of the input image is represented by a 1-channel, 8-bit, unsigned integer
Each pixel of the output image is represented by a 1-channel, 8-bit, unsigned integer
The images will be passed in and out as single-row arrays. Defined constants for 'HEIGHT' and 'WIDTH' (both set to 128) should be used to correctly break the array into a 2D image.
Implementation details should match the OpenCV implementation unless otherwise specified

void bad_pixel_correction_opencv((uint8_t*)img_in, (uint8_t*)img_out);
"""
)


desired_test = "auto_white_balance_opencv"
desired_test_description = (
    """
This algorithm scales the values of pixels based on a gray-world assumption which states that the average of all channels should result in a gray image. It adds a modification which thresholds pixels based on their saturation value and only uses pixels below the provided threshold in finding average pixel values. Saturation is calculated using the following for a 3-channel RGB image per pixel I and is in the range [0, 1]:

Saturation(I) = (max(R,G,B)-min(R,G,B))/max(R,G,B)

A threshold of 1 means that all pixels are used to white-balance, while a threshold of 0 means no pixels are used. Lower thresholds are useful in white-balancing saturated images.

A simple white balance algorithm that works by independently stretching each of the input image channels to the specified range(maximum and minimum). Computes channel wise intensity histogram and ignores p% maximum and minimum values and finally normalize each channel with min and max. For increased robustness it ignores the top and bottom p% (4% is fixed) of pixel values.

The input image will be 3840x2160 pixels
Each pixel of the output image is represented by a 3-channel, 8-bit, unsigned integer
Each pixel of the output image is represented by a 3-channel, 8-bit, unsigned integer
The image will be passed in and out as single-row arrays. Defined constants for 'HEIGHT' and 'WIDTH' should be used to correctly break the array into a 2D image.
Implementation details should match the OpenCV implementation unless otherwise specified

void auto_white_balance_opencv((uint8_t*)in_img, (uint8_t*)out_img,
                               (float)thresh, (float)inputMin, (float)inputMax, (float)outputMin, (float)outputMax);
    """
)


desired_test = "auto_exposure_correction_opencv"
desired_test_description = (
    """
Auto exposure correction improves contrast and brightness of the image and also corrects the exposure of the input frame. The algorithm uses luminence histogram equalization to improve overall exposure and contrast of the image. Luminence (V) is extracted after converting input image to HSV color space. Once the algorthm is applied the image is converted back to RGB color space.

The input image will be 128x128 pixels
Each pixel of the output image is represented by a 3-channel, 8-bit, unsigned integer
Each pixel of the output image is represented by a 3-channel, 8-bit, unsigned integer
The image will be passed in and out as single-row arrays. Defined constants for 'HEIGHT' and 'WIDTH' (both set to 128) should be used to correctly break the array into a 2D image.
Implementation details should match the OpenCV implementation unless otherwise specified

void auto_exposure_correction_opencv((uint8_t*)in_img.data, (uint8_t*)out_img_hls.data);
    """
)


desired_test = "add_weighted_opencv"
desired_test_description = (
    """
The addweighted function calculates a weighted sum of two input images src1, src2 and generates the result in dst.


dst[i] = alpha * src1[i] + beta * src2[i] + gamma

dst[i] is the i-th element of the output array.
src1[i] is the i-th element of the first input array.
src2[i] is the i-th element of the second input array.


The input image will be 128x128 pixels
Each pixel of the output image is represented by a 1-channel, 8-bit, unsigned integer
Each pixel of the output image is represented by a 1-channel, 8-bit, unsigned integer
The image will be passed in and out as single-row arrays. Defined constants for 'HEIGHT' and 'WIDTH' (both set to 128) should be used to correctly break the array into a 2D image.
Implementation details should match the OpenCV Reduce implementation unless otherwise specified

void add_weighted_opencv((uint8_t*)in_gray.data, (uint8_t*)in_gray1.data, (uint8_t*)dst_hls.data, float(alpha), float(beta), float(gamma));
    """
)


desired_test = "accumulate_weighted_opencv"
desired_test_description = (
    """
The accumulate weighted function computes the weighted sum of the input image (src1) and the accumulator image (src2) and generates the result in dst.

dst[i] = alpha * src1[i] + (1-alpha) * src2[i]

dst[i] is the i-th element of the output array.
src1[i] is the i-th element of the first input array.
src2[i] is the i-th element of the second input array.


The input image will be 128x128 pixels
Each pixel of the output image is represented by a 1-channel, 8-bit, unsigned integer
Each pixel of the output image is represented by a 1-channel, 16-bit, unsigned integer
The image will be passed in and out as single-row arrays. Defined constants for 'HEIGHT' and 'WIDTH' (both set to 128) should be used to correctly break the array into a 2D image.
Implementation details should match the OpenCV Reduce implementation unless otherwise specified

void accumulate_weighted_opencv((uint8_t*)in_gray.data, (uint8_t*)in_gray1.data, (uint16_t*)dst_hls.data, float(alpha));
    """
)


desired_test = "accumulate_squared_opencv"
desired_test_description = (
    """
The accumulate square function adds the square of an image (src1) to the accumulator image (src2) and generates the accumulated result (dst).

dst[i] = src1[i]^2 + src2[i]

dst[i] is the i-th element of the output array.
src1[i] is the i-th element of the first input array.
src2[i] is the i-th element of the second input array.


The input image will be 128x128 pixels
Each pixel of the output image is represented by a 1-channel, 8-bit, unsigned integer
Each pixel of the output image is represented by a 1-channel, 16-bit, unsigned integer
The image will be passed in and out as single-row arrays. Defined constants for 'HEIGHT' and 'WIDTH' (both set to 128) should be used to correctly break the array into a 2D image.
Implementation details should match the OpenCV Reduce implementation unless otherwise specified

void accumulate_squared_opencv(uint8_t*)in_gray.data, (uint8_t*)in_gray1.data, (uint16_t*)dst_hls.data);
    """
)


desired_test = "axpy_int32_res"
desired_test_description = (
    """
The axpy function multiplies an input vector (X) of type int32 by an alpha value of the same type. Then it adds the second vector (Y) withe the same type.
The output is the sum of these vectors in vector (Y_res)

Y_res = alpha*X + Y

The input vectors will be 4096 values long
    """
)

desired_test = "axpy_int32"
desired_test_description = (
    """
The axpy function multiplies an input vector (X) of type int32 by an alpha value of the same type. Then it adds the second vector (Y) withe the same type.
The output is the sum of these vectors in vector (Y)

Y = alpha*X + Y

The input vectors will be 4096 values long
    """
)


desired_test = "scale_vector_uint16"
desired_test_description = (
    """
The scale vector function multiplies each element of 16 bit unsigned array (input) by some scalar value (alpha), and generates the resulting array (output)

output[i] = input[i] * alpha

output[i] is the i-th element of the output array.
input[i] is the i-th element of the input array.
alpha is the scalar value to be multiplied.

The arrays will have 8192 unsigned 16 bit integers. 
    """
)



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

void accumulate_opencv(uint8_t*)in_gray.data, (uint8_t*)in_gray1.data, (uint16_t*)dst_hls.data);
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