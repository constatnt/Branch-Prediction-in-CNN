#include "common.h"
#include "image_fully.h"

int max_element(float element [1][10])
{
	int i;
	float max;
	int index=0;

	max = element[0][0];
	for (i = 1; i < 10; i++)
	{
		if (max < element[0][i])
		{
			max = element[0][i];
			index = i;
		}
		else
			max = max;
	}

	//printf("max is %f\n", max);
	return index;
}
void CLASSIFY(float NN_INPUT[10][120], 
			  float Weight_1[120][84], float Bias_1[84], 
			  float Weight_2[84][10], float Bias_2[84],
			  cl_context context, cl_command_queue cmdqueue,
			  cl_kernel FullyConnected_1, cl_kernel FullyConnected_2,
			  unsigned char LABEL[10])
{
	// Hidden Layer Parameter
	float Hidden_Layer[10][84] = { 0, };
	// Output Node
	float Output_NN[10][10] = { 0, };

	// OpenCL for Fully Connected  Layer 1
	// ======================================================================================================
	int clerr;

	size_t fc1_src_size = sizeof(float) * INPUT_NN_1_SIZE * image_Batch;
	size_t fc1_filter_size = sizeof(float) * FILTER_NN_1_SIZE;
	size_t fc1_bias_size = sizeof(float) *  BIAS_NN_1_SIZE;
	size_t fc1_dst_size = sizeof(float) * BIAS_NN_1_SIZE * image_Batch;

	// Cl_mem Object 
	// ------------------------------------------------------------------------------------------------------
	cl_mem cl_fc1_src = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, fc1_src_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_fc1_filter = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, fc1_filter_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_fc1_bias = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, fc1_bias_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_fc1_dst = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY, fc1_dst_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}


	// kernel Argument
	// ------------------------------------------------------------------------------------------------------
	clerr = clSetKernelArg(FullyConnected_1, 0, sizeof(cl_mem), (void *)&cl_fc1_src);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg0 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(FullyConnected_1, 1, sizeof(cl_mem), (void *)&cl_fc1_filter);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg1 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(FullyConnected_1, 2, sizeof(cl_mem), (void *)&cl_fc1_bias);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg2 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(FullyConnected_1, 3, sizeof(cl_mem), (void *)&cl_fc1_dst);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg2 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	// Write Buffer
	// ------------------------------------------------------------------------------------------------------
	clerr = clEnqueueWriteBuffer(cmdqueue, cl_fc1_src, CL_TRUE, 0, fc1_src_size, (void *)NN_INPUT, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clEnqueueWriteBuffer(cmdqueue, cl_fc1_filter, CL_TRUE, 0, fc1_filter_size, (void *)Weight_1, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clEnqueueWriteBuffer(cmdqueue, cl_fc1_bias, CL_TRUE, 0, fc1_bias_size, (void *)Bias_1, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	// Enqueue NDRange
	// ------------------------------------------------------------------------------------------------------

	size_t  GlobalWorkSize[3];
	GlobalWorkSize[0] = image_Batch;
	GlobalWorkSize[1] = INPUT_NN_2_SIZE;
	//GlobalWorkSize[2] = CONV_1_OUTPUT_WH;

	//depth_out = get_global_id(0);
	//row = get_global_id(1);
	//col = get_global_id(2);

	size_t  LocalWorkSize[2];
	LocalWorkSize[0] = 5;
	LocalWorkSize[1] = 5;

	clerr = clEnqueueNDRangeKernel(cmdqueue, FullyConnected_1, 2, NULL, GlobalWorkSize, NULL, 0, NULL, NULL);
	if (clerr != CL_SUCCESS) {
		printf("ERROR :clEnqueueNDRangeKernel-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}
	clerr = clEnqueueReadBuffer(cmdqueue, cl_fc1_dst, CL_TRUE, 0, fc1_dst_size, (void *)Hidden_Layer, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueReadBuffer - cl_Conv_2_buff, [%d]\n", clerr);
		exit(1);
	}


	//fully_connected_0(NN_INPUT, Weight_1, Bias_1, Hidden_Layer);


	#ifdef LOG_PRINT 
	int z, x, y;
		for (z = 0; z < 10; z++)
		{
			for (x = 0; x < 84; x++)
			{
				fprintf(RESULT_FC_1, "%.6f ", Hidden_Layer[z][x]);
			}
			fprintf(RESULT_FC_1, "\n");
		}
	#endif 
	
	// OpenCL for Fully Connected  Layer 2
	// ======================================================================================================
	size_t fc2_src_size = sizeof(float) * INPUT_NN_2_SIZE * image_Batch;
	size_t fc2_filter_size = sizeof(float) * FILTER_NN_2_SIZE;
	size_t fc2_bias_size = sizeof(float) *  BIAS_NN_2_SIZE;
	size_t fc2_dst_size = sizeof(float) * OUTPUT_NN_2_SIZE * image_Batch;

	// Cl_mem Object 
	// ------------------------------------------------------------------------------------------------------
	cl_mem cl_fc2_src = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, fc2_src_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_fc2_filter = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, fc2_filter_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_fc2_bias = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, fc2_bias_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer - cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	cl_mem cl_fc2_dst = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY, fc2_dst_size, NULL, &clerr);
	if (clerr != CL_SUCCESS) {
		printf("ERROR : clCreateBuffer -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}


	// kernel Argument
	// ------------------------------------------------------------------------------------------------------
	clerr = clSetKernelArg(FullyConnected_2, 0, sizeof(cl_mem), (void *)&cl_fc2_src);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg0 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}
	
	clerr = clSetKernelArg(FullyConnected_2, 1, sizeof(cl_mem), (void *)&cl_fc2_filter);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg1 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(FullyConnected_2, 2, sizeof(cl_mem), (void *)&cl_fc2_bias);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg2 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clSetKernelArg(FullyConnected_2, 3, sizeof(cl_mem), (void *)&cl_fc2_dst);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR : clSetKernelArg2 -cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	// Write Buffer
	// ------------------------------------------------------------------------------------------------------
	clerr = clEnqueueWriteBuffer(cmdqueue, cl_fc2_src, CL_TRUE, 0, fc2_src_size, (void *)Hidden_Layer, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}
	
	clerr = clEnqueueWriteBuffer(cmdqueue, cl_fc2_filter, CL_TRUE, 0, fc2_filter_size, (void *)Weight_2, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	clerr = clEnqueueWriteBuffer(cmdqueue, cl_fc2_bias, CL_TRUE, 0, fc2_bias_size, (void *)Bias_2, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueWriteBuffer-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}

	// Enqueue NDRange
	// ------------------------------------------------------------------------------------------------------
	
	GlobalWorkSize[0] = image_Batch;
	GlobalWorkSize[1] = OUTPUT_NN_2_SIZE;
	//GlobalWorkSize[2] = CONV_1_OUTPUT_WH;

	//depth_out = get_global_id(0);
	//row = get_global_id(1);
	//col = get_global_id(2);

	LocalWorkSize[0] = 5;
	LocalWorkSize[1] = 5;

	clerr = clEnqueueNDRangeKernel(cmdqueue, FullyConnected_2, 2, NULL, GlobalWorkSize, NULL, 0, NULL, NULL);
	if (clerr != CL_SUCCESS) {
		printf("ERROR :clEnqueueNDRangeKernel-cl1_kernel, [%d]\n", clerr);
		exit(1);
	}
	
	clerr = clEnqueueReadBuffer(cmdqueue, cl_fc2_dst, CL_TRUE, 0, fc2_dst_size, (void *)Output_NN, 0, NULL, NULL);
	if (clerr != CL_SUCCESS)
	{
		printf("ERROR :clEnqueueReadBuffer - cl_Conv_2_buff, [%d]\n", clerr);
		exit(1);
	}


	//Fully Connected Layer 2
	//fully_connected_1(Hidden_Layer, Weight_2, Bias_2, Output_NN);
	

	#ifdef LOG_PRINT 
	for (z = 0; z < 10; z++)
	{
		for (x = 0; x < 10; x++)
		{
			fprintf(RESULT_FC_2, "%.6f ", Output_NN[z][x]);
		}
		fprintf(RESULT_FC_2, "\n");
	}
	#endif 

	for (int j = 0; j < 10; j++)
	{
		if (LABEL[j] == max_element(Output_NN[j]))
			total_cnt++;
		
	}

}
