#include "common.h"
#include "image_convolution.h"
#include "image_pool.h"

void FEATURE_EXTRACTION(float *SRC,
	float CONV_1_FILTER[6][25], float CONV_1_BIAS[6],
	float POOL_1_FILTER[6][4], float POOL_1_BIAS[6],
	float CONV_2_FILTER[16][6][25], float CONV_2_BIAS[16],
	float POOL_2_FILTER[16][4], float POOL_2_BIAS[6],
	float CONV_3_FILTER[120][16][25], float CONV_3_BIAS[120],
	float *DST,
	cl_context context,
	cl_kernel conv_1_kernel, cl_kernel pool_1_kernel,
	cl_kernel conv_2_kernel, cl_kernel pool_2_kernel,
	cl_kernel conv_3_kernel,
	cl_command_queue cmdqueue
)
{
	int z, x, y;
	
	float *INPUT_FEATURE_MAP = calloc(image_Batch *INPUT_WH * INPUT_WH, sizeof(float));
	float *CONV_1_RESULT = calloc(image_Batch * CONV_1_TYPE * MNIST_SIZE, sizeof(float));
	float *POOL_1_RESULT = calloc(image_Batch * CONV_1_TYPE * POOL_1_OUTPUT_SIZE, sizeof(float));
	float *CONV_2_RESULT = calloc(image_Batch * CONV_2_TYPE * CONV_2_OUTPUT_SIZE, sizeof(float));
	float *POOL_2_RESULT = calloc(image_Batch * CONV_2_TYPE * POOL_2_OUTPUT_SIZE, sizeof(float));
	float *CONV_3_RESULT = calloc(image_Batch * CONV_3_TYPE , sizeof(float));
	
	// 1D-Array to 2D-Array
	for (z = 0; z < image_Batch; z++)
	{
		for (x = 0; x < 32; x++)
		{
			for (y = 0; y < 32; y++)
			{
				INPUT_FEATURE_MAP[32 * 32 * z + 32 * x + y] = SRC[32 * 32 * z + 32 * x + y];
			}
		}
	}

	// OpenCL for Convolution Layer 1
	// ======================================================================================================
	int clerr;

	size_t conv1_src_size = sizeof(float) * INPUT_SIZE * image_Batch;
	size_t conv1_filter_size = sizeof(float) * CONV_1_WH * CONV_1_WH * CONV_1_TYPE;
	size_t conv1_bias_size = sizeof(float) *  CONV_1_TYPE;
	size_t conv1_dst_size = sizeof(float) * CONV_1_OUTPUT_WH * CONV_1_OUTPUT_WH * CONV_1_TYPE * image_Batch;

	// Cl_mem Object 
	// ------------------------------------------------------------------------------------------------------
	cl_mem cl_conv1_src = clCreateBuffer(context,  CL_MEM_READ_ONLY, conv1_src_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_conv1_filter = clCreateBuffer(context,  CL_MEM_READ_ONLY, conv1_filter_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_conv1_bias = clCreateBuffer(context,  CL_MEM_READ_ONLY, conv1_bias_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_conv1_dst = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, conv1_dst_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	// kernel Argument
	// ------------------------------------------------------------------------------------------------------
	clerr = clSetKernelArg(conv_1_kernel, 0, sizeof(cl_mem), (void *)&cl_conv1_src);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg0 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(conv_1_kernel, 1, sizeof(cl_mem), (void *)&cl_conv1_filter);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg1 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(conv_1_kernel, 2, sizeof(cl_mem), (void *)&cl_conv1_bias);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg2 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(conv_1_kernel, 3, sizeof(cl_mem), (void *)&cl_conv1_dst);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg2 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	// Write Buffer
	// ------------------------------------------------------------------------------------------------------
	clerr = clEnqueueWriteBuffer(cmdqueue, cl_conv1_src, CL_TRUE, 0, conv1_src_size, (void *)INPUT_FEATURE_MAP, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clEnqueueWriteBuffer(cmdqueue, cl_conv1_filter, CL_TRUE, 0, conv1_filter_size, (void *)CONV_1_FILTER, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clEnqueueWriteBuffer(cmdqueue, cl_conv1_bias, CL_TRUE, 0, conv1_bias_size, (void *)CONV_1_BIAS, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	// Enqueue NDRange
	// ------------------------------------------------------------------------------------------------------

	size_t  GlobalWorkSize[3];
	
	GlobalWorkSize[0] = image_Batch;
	GlobalWorkSize[1] = CONV_1_OUTPUT_WH * CONV_1_OUTPUT_WH;
	GlobalWorkSize[2] = CONV_1_TYPE;
	
	size_t LocalWorkSize[3];

	clerr = clEnqueueNDRangeKernel(cmdqueue, conv_1_kernel,3, NULL, GlobalWorkSize, NULL, 0, NULL, NULL);
	if (clerr != CL_SUCCESS) {
		printf("ERROR :clEnqueueNDRangeKernel-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}
	clerr = clEnqueueReadBuffer(cmdqueue, cl_conv1_dst, CL_TRUE, 0, conv1_dst_size, (void *)CONV_1_RESULT, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueReadBuffer - cl_Conv_2_buff, [%d]\n", clerr);
		exit(1);
	}

	CONVOLUTION_LAYER_1(INPUT_FEATURE_MAP, CONV_1_FILTER, CONV_1_BIAS, CONV_1_RESULT);

	#ifdef LOG_PRINT 
		for (z = 0; z < 60; z++)
		{
			for (x = 0; x < 28; x++)
			{
				for (y = 0; y < 28; y++)
				{
					fprintf(RESULT_CONV_1, "%.10f ", CONV_1_RESULT[z *28*28+ 28 * x + y]);
				}
				fprintf(RESULT_CONV_1, "\n");
			}
			fprintf(RESULT_CONV_1, "\n");
		}
	#endif 

	// OpenCL for Pooling Layer 1
	// ======================================================================================================
	size_t pool1_src_size = sizeof(float) * POOL_1_INPUT_SIZE * image_Batch * POOL_1_TYPE;
	size_t pool1_filter_size = sizeof(float) * POOL_1_TYPE * POOL_1_SIZE;
	size_t pool1_bias_size = sizeof(float) *  POOL_1_TYPE;
	size_t pool1_dst_size = sizeof(float) * POOL_1_OUTPUT_SIZE * POOL_1_TYPE * image_Batch;

	// Cl_mem Object 
	// ------------------------------------------------------------------------------------------------------
	cl_mem cl_pool1_src = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, pool1_src_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_pool1_filter = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY , pool1_filter_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_pool1_bias = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, pool1_bias_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_pool1_dst = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY, pool1_dst_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}


	// kernel Argument
	// ------------------------------------------------------------------------------------------------------
	clerr = clSetKernelArg(pool_1_kernel, 0, sizeof(cl_mem), (void *)&cl_pool1_src);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg0 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(pool_1_kernel, 1, sizeof(cl_mem), (void *)&cl_pool1_filter);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg1 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(pool_1_kernel, 2, sizeof(cl_mem), (void *)&cl_pool1_bias);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg2 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(pool_1_kernel, 3, sizeof(cl_mem), (void *)&cl_pool1_dst);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg2 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	// Write Buffer
	// ------------------------------------------------------------------------------------------------------
	clerr = clEnqueueWriteBuffer(cmdqueue, cl_pool1_src, CL_TRUE, 0, pool1_src_size, (void *)CONV_1_RESULT, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clEnqueueWriteBuffer(cmdqueue, cl_pool1_filter, CL_TRUE, 0, pool1_filter_size, (void *)POOL_1_FILTER, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clEnqueueWriteBuffer(cmdqueue, cl_pool1_bias, CL_TRUE, 0, pool1_bias_size, (void *)POOL_1_BIAS, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	// Enqueue NDRange
	// ------------------------------------------------------------------------------------------------------
	GlobalWorkSize[0] = image_Batch * POOL_1_OUTPUT_WH;
	GlobalWorkSize[1] = POOL_1_TYPE * POOL_1_OUTPUT_WH;

	clerr = clEnqueueNDRangeKernel(cmdqueue, pool_1_kernel, 2, NULL, GlobalWorkSize, NULL, 0, NULL, NULL);
	if (clerr != CL_SUCCESS) {
		printf("ERROR :clEnqueueNDRangeKernel-pl1_kernel, [%d]\n", clerr);
		exit(1);
	}
	clerr = clEnqueueReadBuffer(cmdqueue, cl_pool1_dst, CL_TRUE, 0, pool1_dst_size, (void *)POOL_1_RESULT, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueReadBuffer - cl_Conv_2_buff, [%d]\n", clerr);
		exit(1);
	}


	//POOLING_LAYER_1(CONV_1_RESULT, POOL_1_FILTER, POOL_1_BIAS, POOL_1_RESULT, 2);


	#ifdef LOG_PRINT 
		for (z = 0; z < 60; z++)
		{
			for (x = 0; x < 14; x++)
			{
				for (y = 0; y < 14; y++)
				{
					fprintf(RESULT_POOL_1, "%.9f ", POOL_1_RESULT[z*14*14+14 * x + y]);
				}
				fprintf(RESULT_POOL_1, "\n");
			}
			fprintf(RESULT_POOL_1, "\n");
		}
	#endif 

	// OpenCL for Convolution Layer 2
	// ======================================================================================================
	//ss = clock();
	size_t conv2_src_size = sizeof(float) * CONV_2_INPUT_SIZE * image_Batch * POOL_1_TYPE;
	size_t conv2_filter_size = sizeof(float) * CONV_2_TYPE * CONV_2_WH * CONV_2_WH * POOL_1_TYPE;
	size_t conv2_bias_size = sizeof(float) *  CONV_2_TYPE;
	size_t conv2_dst_size = sizeof(float) * CONV_2_OUTPUT_SIZE * CONV_2_TYPE * image_Batch;

	// Cl_mem Object 
	// ------------------------------------------------------------------------------------------------------
	cl_mem cl_conv2_src = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, conv2_src_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_conv2_filter = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,  conv2_filter_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_conv2_bias = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, conv2_bias_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_conv2_dst = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY , conv2_dst_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	// kernel Argument
	// ------------------------------------------------------------------------------------------------------
	clerr = clSetKernelArg(conv_2_kernel, 0, sizeof(cl_mem), (void *)&cl_conv2_src);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg0 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(conv_2_kernel, 1, sizeof(cl_mem), (void *)&cl_conv2_filter);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg1 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(conv_2_kernel, 2, sizeof(cl_mem), (void *)&cl_conv2_bias);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg2 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(conv_2_kernel, 3, sizeof(cl_mem), (void *)&cl_conv2_dst);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg2 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	// Write Buffer
	// ------------------------------------------------------------------------------------------------------
	clerr = clEnqueueWriteBuffer(cmdqueue, cl_conv2_src, CL_TRUE, 0, conv2_src_size, (void *)POOL_1_RESULT, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clEnqueueWriteBuffer(cmdqueue, cl_conv2_filter, CL_TRUE, 0, conv2_filter_size, (void *)CONV_2_FILTER, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clEnqueueWriteBuffer(cmdqueue, cl_conv2_bias, CL_TRUE, 0, conv2_bias_size, (void *)CONV_2_BIAS, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	// Enqueue NDRange
	// ------------------------------------------------------------------------------------------------------
	GlobalWorkSize[0] = image_Batch * CONV_2_TYPE;
	GlobalWorkSize[1] = CONV_2_OUTPUT_SIZE;
	
	clerr = clEnqueueNDRangeKernel(cmdqueue, conv_2_kernel, 2, NULL, GlobalWorkSize, NULL, 0, NULL, NULL);
	if (clerr != CL_SUCCESS) {
		printf("ERROR :clEnqueueNDRangeKernel-cl2_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clEnqueueReadBuffer(cmdqueue, cl_conv2_dst, CL_TRUE, 0, conv2_dst_size, (void *)CONV_2_RESULT, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueReadBuffer - cl_Conv_2_buff, [%d]\n", clerr);
		exit(1);
	}
	
	//CONVOLUTION_LAYER_2(POOL_1_RESULT, CONV_2_FILTER, CONV_2_BIAS, CONV_2_RESULT);

	#ifdef LOG_PRINT 
		for (z = 0; z < 160; z++)
		{
			for (x = 0; x < 10; x++)
			{
				for (y = 0; y < 10; y++)
				{
					fprintf(RESULT_CONV_2, "%.9f ", CONV_2_RESULT[z*100+10 * x + y]);
				}
				fprintf(RESULT_CONV_2, "\n");
			}
			fprintf(RESULT_CONV_2, "\n");
		}
	#endif 

	// OpenCL for Pooling Layer 2
	// ======================================================================================================
	size_t pool2_src_size = sizeof(float) * POOL_2_INPUT_SIZE * image_Batch * CONV_2_TYPE;
	size_t pool2_filter_size = sizeof(float) * CONV_2_TYPE * POOL_1_SIZE;
	size_t pool2_bias_size = sizeof(float) *  CONV_2_TYPE;
	size_t pool2_dst_size = sizeof(float) * POOL_2_OUTPUT_SIZE * CONV_2_TYPE * image_Batch;

	// Cl_mem Object 
	// ------------------------------------------------------------------------------------------------------
	cl_mem cl_pool2_src = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, pool2_src_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_pool2_filter = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, pool2_filter_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_pool2_bias = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, pool2_bias_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_pool2_dst = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY, pool2_dst_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}


	// kernel Argument
	// ------------------------------------------------------------------------------------------------------
	clerr = clSetKernelArg(pool_2_kernel, 0, sizeof(cl_mem), (void *)&cl_pool2_src);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg0 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(pool_2_kernel, 1, sizeof(cl_mem), (void *)&cl_pool2_filter);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg1 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(pool_2_kernel, 2, sizeof(cl_mem), (void *)&cl_pool2_bias);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg2 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(pool_2_kernel, 3, sizeof(cl_mem), (void *)&cl_pool2_dst);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg2 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}


	// Write Buffer
	// ------------------------------------------------------------------------------------------------------
	clerr = clEnqueueWriteBuffer(cmdqueue, cl_pool2_src, CL_TRUE, 0, pool2_src_size, (void *)CONV_2_RESULT, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clEnqueueWriteBuffer(cmdqueue, cl_pool2_filter, CL_TRUE, 0, pool2_filter_size, (void *)POOL_2_FILTER, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clEnqueueWriteBuffer(cmdqueue, cl_pool2_bias, CL_TRUE, 0, pool2_bias_size, (void *)POOL_2_BIAS, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	// Enqueue NDRange
	// ------------------------------------------------------------------------------------------------------
	GlobalWorkSize[0] = image_Batch * POOL_2_OUTPUT_WH;
	GlobalWorkSize[1] = POOL_2_TYPE * POOL_2_OUTPUT_WH;

	clerr = clEnqueueNDRangeKernel(cmdqueue, pool_2_kernel, 2, NULL, GlobalWorkSize, NULL, 0, NULL, NULL);
	if (clerr != CL_SUCCESS) {
		printf("ERROR :clEnqueueNDRangeKernel-cl2_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clEnqueueReadBuffer(cmdqueue, cl_pool2_dst, CL_TRUE, 0, pool2_dst_size, (void *)POOL_2_RESULT, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueReadBuffer - cl_Conv_2_buff, [%d]\n", clerr);
		exit(1);
	}

	#ifdef LOG_PRINT 
		for (z = 0; z < 160; z++)
		{
			for (x = 0; x < 5; x++)
			{
				for (y = 0; y < 5; y++)
				{
					fprintf(RESULT_POOL_2, "%.6f ", POOL_2_RESULT[z*25+5 * x + y]);
				}
				fprintf(RESULT_POOL_2, "\n");
			}
			fprintf(RESULT_POOL_2, "\n");
		}
	#endif 

	// OpenCL for Convolution Layer 3
	// ======================================================================================================
	//ss = clock();
	size_t conv3_src_size = sizeof(float) * CONV_3_INPUT_SIZE * image_Batch * CONV_2_TYPE;
	size_t conv3_filter_size = sizeof(float) * CONV_3_TYPE * CONV_3_WH * CONV_3_WH * CONV_2_TYPE;
	size_t conv3_bias_size = sizeof(float) *  CONV_3_TYPE;
	size_t conv3_dst_size = sizeof(float) * CONV_3_OUTPUT_SIZE * CONV_3_TYPE * image_Batch;

	// Cl_mem Object 
	// ------------------------------------------------------------------------------------------------------
	cl_mem cl_conv3_src = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, conv3_src_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_conv3_filter = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, conv3_filter_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_conv3_bias = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, conv3_bias_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_conv3_dst = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY, conv3_dst_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	// kernel Argument
	// ------------------------------------------------------------------------------------------------------
	clerr = clSetKernelArg(conv_3_kernel, 0, sizeof(cl_mem), (void *)&cl_conv3_src);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg0 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(conv_3_kernel, 1, sizeof(cl_mem), (void *)&cl_conv3_filter);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg1 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(conv_3_kernel, 2, sizeof(cl_mem), (void *)&cl_conv3_bias);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg2 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(conv_3_kernel, 3, sizeof(cl_mem), (void *)&cl_conv3_dst);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg2 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	// Write Buffer
	// ------------------------------------------------------------------------------------------------------
	clerr = clEnqueueWriteBuffer(cmdqueue, cl_conv3_src, CL_TRUE, 0, conv3_src_size, (void *)POOL_2_RESULT, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clEnqueueWriteBuffer(cmdqueue, cl_conv3_filter, CL_TRUE, 0, conv3_filter_size, (void *)CONV_3_FILTER, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clEnqueueWriteBuffer(cmdqueue, cl_conv3_bias, CL_TRUE, 0, conv3_bias_size, (void *)CONV_3_BIAS, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	// Enqueue NDRange
	// ------------------------------------------------------------------------------------------------------
	GlobalWorkSize[0] = image_Batch;
	GlobalWorkSize[1] = CONV_3_TYPE;
	GlobalWorkSize[2] = 0;

	//depth_out = get_global_id(0);
	//row = get_global_id(1);
	//col = get_global_id(2);

	LocalWorkSize[0] = 1;
	LocalWorkSize[1] = 1;
	LocalWorkSize[2] = 1;

	clerr = clEnqueueNDRangeKernel(cmdqueue, conv_3_kernel, 2, NULL, GlobalWorkSize, NULL, 0, NULL, NULL);
	if (clerr != CL_SUCCESS) {
		printf("ERROR :clEnqueueNDRangeKernel-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clEnqueueReadBuffer(cmdqueue, cl_conv3_dst, CL_TRUE, 0, conv3_dst_size, (void *)DST, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueReadBuffer - cl_Conv_2_buff, [%d]\n", clerr);
		exit(1);
	}

	//CONVOLUTION_LAYER_3(POOL_2_RESULT, CONV_3_FILTER, CONV_3_BIAS, DST);

	#ifdef LOG_PRINT 
		for (z = 0; z < 10; z++)
		{
			for (x = 0; x < 120; x++)
			{
				fprintf(RESULT_CONV_3, "%.6f\n ", DST[z * 120 + x]);
			}
			fprintf(RESULT_CONV_3, "\n");
			fprintf(RESULT_CONV_3, "\n");
		}
	#endif 
		
		
	free(INPUT_FEATURE_MAP);
	free(CONV_1_RESULT);
	free(POOL_1_RESULT);
	free(CONV_2_RESULT);
	free(POOL_2_RESULT);
	free(CONV_3_RESULT);
	
	}