/* Udacity Homework 3
   HDR Tone-mapping

   Background HDR
   ==============

   A High Dynamic Range (HDR) image contains a wider variation of intensity
   and color than is allowed by the RGB format with 1 byte per channel that we
   have used in the previous assignment.  

   To store this extra information we use single precision floating point for
   each channel.  This allows for an extremely wide range of intensity values.

   In the image for this assignment, the inside of church with light coming in
   through stained glass windows, the raw input floating point values for the
   channels range from 0 to 275.  But the mean is .41 and 98% of the values are
   less than 3!  This means that certain areas (the windows) are extremely bright
   compared to everywhere else.  If we linearly map this [0-275] range into the
   [0-255] range that we have been using then most values will be mapped to zero!
   The only thing we will be able to see are the very brightest areas - the
   windows - everything else will appear pitch black.

   The problem is that although we have cameras capable of recording the wide
   range of intensity that exists in the real world our monitors are not capable
   of displaying them.  Our eyes are also quite capable of observing a much wider
   range of intensities than our image formats / monitors are capable of
   displaying.

   Tone-mapping is a process that transforms the intensities in the image so that
   the brightest values aren't nearly so far away from the mean.  That way when
   we transform the values into [0-255] we can actually see the entire image.
   There are many ways to perform this process and it is as much an art as a
   science - there is no single "right" answer.  In this homework we will
   implement one possible technique.

   Background Chrominance-Luminance
   ================================

   The RGB space that we have been using to represent images can be thought of as
   one possible set of axes spanning a three dimensional space of color.  We
   sometimes choose other axes to represent this space because they make certain
   operations more convenient.

   Another possible way of representing a color image is to separate the color
   information (chromaticity) from the brightness information.  There are
   multiple different methods for doing this - a common one during the analog
   television days was known as Chrominance-Luminance or YUV.

   We choose to represent the image in this way so that we can remap only the
   intensity channel and then recombine the new intensity values with the color
   information to form the final image.

   Old TV signals used to be transmitted in this way so that black & white
   televisions could display the luminance channel while color televisions would
   display all three of the channels.


   Tone-mapping
   ============

   In this assignment we are going to transform the luminance channel (actually
   the log of the luminance, but this is unimportant for the parts of the
   algorithm that you will be implementing) by compressing its range to [0, 1].
   To do this we need the cumulative distribution of the luminance values.

   Example
   -------

input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
min / max / range: 0 / 9 / 9

histo with 3 bins: [4 7 3]

cdf : [4 11 14]


Your task is to calculate this cumulative distribution by following these
steps.

*/

#include "utils.h"

#define BLOCK_SIZE 32

        __device__
		float d_min(float a, float b)
{
		        return (a < b) ? a : b;
}

        __device__
		float d_max(float a, float b)
{
		        return (a > b) ? a : b;
}


		__global__
void find_max_min(const float* const d_logLuminance,
				float* const d_out,
				const bool minMode)
{
		extern __shared__ float s_data[];
		const int myId = blockIdx.x * blockDim.x + threadIdx.x;
		const int tId = threadIdx.x;

		s_data[tId] = d_logLuminance[myId];
		__syncthreads();

		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
		{
			if (tId < s)
			{
				if (minMode)
					s_data[tId] = d_min(s_data[tId], s_data[tId + s]);
				else
					s_data[tId] = d_max(s_data[tId], s_data[tId + s]);
			}	
			__syncthreads();
		}
		if (tId == 0)
			d_out[blockIdx.x] = s_data[tId];
}	

__global__
void histo(const float* const d_logLuminance,
		   unsigned int* const d_histo,
		   const int numBins,
		   const float logLumRange,
		   const float min_logLum)
{
	const int idx1d = blockIdx.x * blockDim.x + threadIdx.x;
	const float logLum = d_logLuminance[idx1d];
	
	unsigned int binPos = static_cast<unsigned int>((logLum - min_logLum) / logLumRange * static_cast<float>(numBins));
	if (binPos >= numBins)
			binPos = numBins - 1;
	atomicAdd(d_histo + binPos, 1);
}

__global__
void hillis_steele_exclusive_scan(unsigned int* const d_cdf,
		  int numBins)
{
	// Hillis-Steele algorihtm
	int idx = threadIdx.x;
	for (int s = 1; s < numBins; s <<= 1)
	{
		if (idx - s >= 0)
			d_cdf[idx] = d_cdf[idx] + d_cdf[idx - s];
		__syncthreads();
	}
	
	if (idx == 0)
	{
		for (int i = numBins - 1; i > 0; --i)
			d_cdf[i] = d_cdf[i - 1];
		d_cdf[0] = 0;
	}
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
				unsigned int* const d_cdf,
				float &min_logLum,
				float &max_logLum,
				const size_t numRows,
				const size_t numCols,
				const size_t numBins)
{
		//TODO
		/*Here are the steps you need to implement
		  1) find the minimum and maximum value in the input logLuminance channel
		  store in min_logLum and max_logLum
		  2) subtract them to find the range
		  3) generate a histogram of all the values in the logLuminance channel using
		  the formula: bin = (lum[i] - lumMin) / lumRange * numBins
		  4) Perform an exclusive scan (prefix sum) on the histogram to get
		  the cumulative distribution of luminance values (this should go in the
		  incoming d_cdf pointer which already has been allocated for you)       */

		const size_t lumSize = numRows * numCols; 
		const dim3 blockSize = dim3(1024);
		const dim3 gridSize = dim3((lumSize + 1024 - 1) / 1024);

		min_logLum = 1.f;
		max_logLum = 0.f;

		float *d_out_min;
		float *d_out_max;
		checkCudaErrors(cudaMalloc(&d_out_min, lumSize * sizeof (float)));
		checkCudaErrors(cudaMalloc(&d_out_max, lumSize * sizeof (float)));

		find_max_min<<<gridSize, blockSize, 1024 * sizeof (float)>>>(d_logLuminance,
						d_out_min,
						true);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		find_max_min<<<gridSize, blockSize, 1024 * sizeof (float)>>>(d_logLuminance,
						d_out_max,
						false);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		find_max_min<<<1, blockSize, 1024 * sizeof (float)>>>(d_out_min,
						d_out_min,
						true);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		find_max_min<<<1, blockSize, 1024 * sizeof (float)>>>(d_out_max,
						d_out_max,
						false);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		checkCudaErrors(cudaMemcpy(&min_logLum, d_out_min, sizeof (float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&max_logLum, d_out_max, sizeof (float), cudaMemcpyDeviceToHost));

		const float logLum_range = max_logLum - min_logLum;

		std::cout << "logLum_range " << logLum_range << std::endl;

		unsigned int* d_bin;
		checkCudaErrors(cudaMalloc(&d_bin, numBins * sizeof (unsigned int)));
		checkCudaErrors(cudaMemset(d_bin, 0, numBins * sizeof (unsigned int)));
	
		histo<<<gridSize, blockSize>>>(d_logLuminance,
						d_bin,
						numBins,
						logLum_range,
						min_logLum);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


		checkCudaErrors(cudaMemcpy(d_cdf, d_bin, numBins * sizeof (unsigned int), cudaMemcpyDeviceToDevice));

		hillis_steele_exclusive_scan<<<1, blockSize>>>(d_cdf, static_cast<int>(numBins));

		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		
		checkCudaErrors(cudaFree(d_out_min));
		checkCudaErrors(cudaFree(d_out_max));
		checkCudaErrors(cudaFree(d_bin));
}
