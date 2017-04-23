// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//
// Writing code the safe way requires slightly more typing, but is very helpful for
// catching mistakes. If you write code the unsafe way and you make a mistake, then
// any subsequent kernels won't compute anything, and it will be hard to figure out
// why. Writing code the safe way will inform you as soon as you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************

#include "utils.h"

		__device__
int d_min(int a, int b)
{
		return (a < b) ? a : b;
}

		__device__
int d_max(int a, int b)
{
		return (a > b) ? a : b;
}

#include<stdio.h>


		__global__
void gaussian_blur(const unsigned char* const inputChannel,
				unsigned char* const outputChannel,
				int numRows, int numCols,
				const float* const filter, const int filterWidth)
{
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= numCols || y >= numRows)
				return;

		float newValue = 0.f;
		for (int i = -filterWidth / 2; i <= filterWidth / 2; ++i)
		{
				const int new_x = d_min(d_max(x + i, 0), numCols - 1);
				for (int j = -filterWidth / 2; j <= filterWidth / 2; ++j)
				{
						const int new_y = d_min(d_max(y + j, 0), numRows - 1);

						float filter_value = filter[(i + filterWidth / 2) * filterWidth + j + (filterWidth / 2)];
						float image_value = static_cast<float>(inputChannel[new_y * numCols + new_x]);	
						newValue = newValue + filter_value * image_value;
				}
		}
		const int thread1dpos = y * numCols + x;

		outputChannel[thread1dpos] = static_cast<unsigned char>(newValue);
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
		__global__
void separateChannels(const uchar4* const inputImageRGBA,
				int numRows,
				int numCols,
				unsigned char* const redChannel,
				unsigned char* const greenChannel,
				unsigned char* const blueChannel)
{
		const int2 thread2dpos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
						blockIdx.y * blockDim.y + threadIdx.y);

		if (thread2dpos.x >= numCols || thread2dpos.y >= numRows)
				return;

		const int thread1dpos = thread2dpos.y * numCols + thread2dpos.x;
		uchar4 rgba = inputImageRGBA[thread1dpos];
		redChannel[thread1dpos] = rgba.x;
		greenChannel[thread1dpos] = rgba.y;
		blueChannel[thread1dpos] = rgba.z;
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
		__global__
void recombineChannels(const unsigned char* const redChannel,
				const unsigned char* const greenChannel,
				const unsigned char* const blueChannel,
				uchar4* const outputImageRGBA,
				int numRows,
				int numCols)
{
		const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
						blockIdx.y * blockDim.y + threadIdx.y);

		const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

		//make sure we don't try and access memory outside the image
		//by having any threads mapped there return early
		if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
				return;

		unsigned char red   = redChannel[thread_1D_pos];
		unsigned char green = greenChannel[thread_1D_pos];
		unsigned char blue  = blueChannel[thread_1D_pos];

		//Alpha should be 255 for no transparency
		uchar4 outputPixel = make_uchar4(red, green, blue, 255);

		outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
				const float* const h_filter, const size_t filterWidth)
{

		checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
		checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
		checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

		size_t filter_byte_size = sizeof(float) * filterWidth * filterWidth;

		checkCudaErrors(cudaMalloc(&d_filter, filter_byte_size));


		float total = 0.f;
		for (int filter_r = - ((int)filterWidth / 2) ; filter_r <= (int)filterWidth / 2; ++filter_r)
		{
				for (int filter_c = - ((int)filterWidth/2); filter_c <= (int)filterWidth/2; ++filter_c) 
				{

						float filter_value = h_filter[(filter_r + (int)filterWidth/2) * filterWidth + filter_c + (int)filterWidth/2];
						total += filter_value;
						std::cout << filter_value << " ";

				}
		}
		std::cout << " = " << total << std::endl;
		std::cout << "Filter width " << filterWidth << " Total value " << total << std::endl;


		checkCudaErrors(cudaMemcpy(d_filter,
								h_filter,
								filter_byte_size,
								cudaMemcpyHostToDevice));
}

#define BLOCK_SIZE 32

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
				uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
				unsigned char *d_redBlurred, 
				unsigned char *d_greenBlurred, 
				unsigned char *d_blueBlurred,
				const int filterWidth)
{

		// Threads per block
		const dim3 blockSize =  dim3(BLOCK_SIZE, BLOCK_SIZE);

		// Grid size
		const dim3 gridSize = dim3(numCols / BLOCK_SIZE + 1, numRows / BLOCK_SIZE + 1);

		//TODO: Launch a kernel for separating the RGBA image into different color channels
		separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,
						numRows,
						numCols,
						d_red,
						d_green,
						d_blue);

		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		gaussian_blur<<<gridSize, blockSize>>>(d_red,
						d_redBlurred,
						numRows,
						numCols,
						d_filter,
						filterWidth);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		gaussian_blur<<<gridSize, blockSize>>>(d_green,
						d_greenBlurred,
						numRows,
						numCols,
						d_filter,
						filterWidth);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		gaussian_blur<<<gridSize, blockSize>>>(d_blue,
						d_blueBlurred,
						numRows,
						numCols,
						d_filter,
						filterWidth);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		// Now we recombine your results. We take care of launching this kernel for you.
		recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
						d_greenBlurred,
						d_blueBlurred,
						d_outputImageRGBA,
						numRows,
						numCols);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}


void cleanup() {
		checkCudaErrors(cudaFree(d_red));
		checkCudaErrors(cudaFree(d_green));
		checkCudaErrors(cudaFree(d_blue));
}
