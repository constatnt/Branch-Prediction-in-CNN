

// Layer Parameter
// ===========================================

// Input Parameter(Batch, MNIST_Set)
#define image_Batch 10
#define MNIST_SIZE 28*28
#define MNIST_WH 28
#define INPUT_SIZE 32*32
#define INPUT_WH 32

// Convolution Layer 1 Parameter
#define CONV_1_OUTPUT_WH 28
#define CONV_1_OUTPUT_SIZE 28*28
#define CONV_1_TYPE 6
#define CONV_1_WH 5

// Pooling Layer 1 Parameter
#define POOL_1_OUTPUT_WH 14
#define POOL_1_OUTPUT_SIZE 14*14
#define POOL_1_INPUT_WH 28
#define POOL_1_INPUT_SIZE 28*28
#define POOL_1_TYPE 6
#define POOL_1_SIZE 4

// Convolution Layer 2 Parameter
#define CONV_2_OUTPUT_WH 10
#define CONV_2_OUTPUT_SIZE 10*10
#define CONV_2_INPUT_WH 14
#define CONV_2_INPUT_SIZE 14*14
#define CONV_2_TYPE 16
#define CONV_2_WH 5

// Pooling Layer 2 Parameter
#define POOL_2_OUTPUT_WH 5
#define POOL_2_OUTPUT_SIZE 5*5
#define POOL_2_INPUT_WH 10
#define POOL_2_INPUT_SIZE 10*10
#define POOL_2_TYPE 16
#define POOL_2_SIZE 4

// Convolution Layer 3 Parameter
#define CONV_3_OUTPUT_WH 1
#define CONV_3_OUTPUT_SIZE 1
#define CONV_3_INPUT_WH 5
#define CONV_3_INPUT_SIZE 5*5
#define CONV_3_TYPE 120
#define CONV_3_WH 5

// Fully Connected Layer Parameter
#define INPUT_NN_1_SIZE 120
#define FILTER_NN_1_SIZE 120 * 84
#define BIAS_NN_1_SIZE 84

#define INPUT_NN_2_SIZE 84
#define OUTPUT_NN_2_SIZE 10

// GB(5, 784, 6)
// Images are organized like [Batch_Index][Row_Index][Col_Index] 
// Filters are organized like [Depth_Out_Index][Row_Index][Col_Index] 
// Output are organized like [Depth_Out_Index][Row_Index][Col_Index] 
// Work size is organized like [Image_Batch][CONV_1_OUTPUT_WH * CONV_1_OUTPUT_WH][CONV_1_TYPE] 
// Make more thread by minumum operations
__kernel void CONVOLUTION_LAYER_1(const __global float * input_feature,
								  const __global float * conv_kernel,
								  const __global float * conv_bias,
								  __global float * output_feature)
{
	int batch_cnt, depth_out;
	int row, col, row_x, col_x;
	int row_f, col_f;
	float  value;

	// Private or Locacl Memory
	float buff_conv_kernel[25];
	float buff_input_feature[25];

	// Work-item index
	batch_cnt = get_global_id(0);
	row = get_global_id(1) / CONV_1_OUTPUT_WH;
	col = get_global_id(1) % CONV_1_OUTPUT_WH;
	depth_out = get_global_id(2);
	
	// Feature Map Start Index
	int src_idx = batch_cnt * INPUT_SIZE;
	int dst_idx = batch_cnt * CONV_1_OUTPUT_SIZE * CONV_1_TYPE ;
	//int conv_idx = depth_out * 25;

	//for (batch_cnt = 0; batch_cnt<image_Batch; batch_cnt++)
	{
		//for (depth_out = 0; depth_out < CONV_1_TYPE; depth_out++)
		{
			//for (row = 0; row < CONV_1_OUTPUT_WH; row++) 
			{
				//for (col = 0; col < CONV_1_OUTPUT_WH; col++)
				{

					// Init
					value = 0;

					if (col == 6 && row == 8 && batch_cnt==0 && depth_out==0)
					{
						//printf("Input image is \n");
						for (row_f = 0; row_f < CONV_1_WH; row_f++) {
							for (col_f = 0; col_f < CONV_1_WH; col_f++) {
								//printf("%.10f\n", input_feature[batch_cnt][INPUT_WH * (row + row_f) + col + col_f]);
								output_feature[CONV_1_OUTPUT_WH * row_f + col_f+ batch_cnt]
									= input_feature[src_idx + (row + row_f) * 32 + (col + col_f)];
							}
						}

						//printf("Convolution kernel is \n");
						for (row_f = 0; row_f < CONV_1_WH; row_f++) {
							for (col_f = 0; col_f < CONV_1_WH; col_f++) {
								//conv_kernel[depth_out][CONV_1_WH * row_f + col_f];
							}
						}
					}


					// Multiplication by Convolution and Input feature map
					for (row_x = 0; row_x < CONV_1_WH; row_x++) {
						for (col_x = 0; col_x < CONV_1_WH; col_x++) {
							//value += input_feature[src_idx + (row + row_x) * 32 + (col + col_x)] * conv_kernel[depth_out * 25 + row_x * 5 + col_x];
						}
					}
					// Result of Convolution 
					//output_feature[dst_idx + depth_out * 28*28 + CONV_1_OUTPUT_WH * row + col] = tanh(value + conv_bias[depth_out]);
				}
			}
		}
	}
}


// Images are organized like [Batch_Index][Depth_In_Index][Row_Index][Col_Index] 
// Filters are organized like [Depth_Out_Index][Depth_In_Index][Row_Index][Col_Index] 
// Output are organized like [Depth_Out_Index][Row_Index][Col_Index] 
// Work SIZE is organized like [image_Batch][POOL_1_OUTPUT_WH*POOL_1_OUTPUT_WH][CONV_1_TYPE] 
__kernel void  POOLING_LAYER_1( __global float * input_feature,
							    __global float *  pool_kernel,
							    __global float *  pool_bias,
							   __global float *  output_feature)
{
	int col, row;
	int depth_out, batch_cnt;
	int row_x, col_x;
	int src_idx, dst_idx;
	float value = 0;

	// Work-item index
	batch_cnt = get_global_id(0) / POOL_1_OUTPUT_WH;
	depth_out = get_global_id(1) / POOL_1_OUTPUT_WH;
	row = get_global_id(0) % POOL_1_OUTPUT_WH;
	col = get_global_id(1) % POOL_1_OUTPUT_WH;

	// Feature Map Start Index
	src_idx = batch_cnt * POOL_1_INPUT_SIZE * POOL_1_TYPE + depth_out * POOL_1_INPUT_SIZE;
	dst_idx = batch_cnt * POOL_1_OUTPUT_SIZE * POOL_1_TYPE + depth_out * POOL_1_OUTPUT_SIZE;
	int conv_idx = depth_out * POOL_1_SIZE;

	// Multiplication by Convolution kernel and Input feautre map
	value = input_feature[src_idx + (row * 2) * POOL_1_INPUT_WH + (col * 2)] * pool_kernel[conv_idx]
	+ input_feature[src_idx + (row * 2) * POOL_1_INPUT_WH + (col * 2 + 1)] * pool_kernel[conv_idx + 1]
	+ input_feature[src_idx + (row * 2 + 1) * POOL_1_INPUT_WH + (col * 2)] * pool_kernel[conv_idx + 2]
	+ input_feature[src_idx + (row * 2 + 1) * POOL_1_INPUT_WH + (col * 2 + 1)] * pool_kernel[conv_idx + 3];
	
	value *= 2.7;

	// Result of Pooling (Activation and Bias)
	//output_feature[dst_idx + row * POOL_1_OUTPUT_WH + col] = tanh(value + pool_bias[depth_out]);
	output_feature[dst_idx + row * POOL_1_OUTPUT_WH + col] =value + pool_bias[depth_out];
}

// Images are organized like [Batch_Index][Depth_In_Index][Row_Index][Col_Index] 
// Filters are organized like [Depth_Out_Index][Depth_In_Index][Row_Index][Col_Index] 
// Output are organized like [Depth_Out_Index][Row_Index][Col_Index] 
// Work size is organized like [Image_Batch][CONV_2_OUTPUT_WH * CONV_2_OUTPUT_WH][CONV_2_TYPE] 
// Make more thread by minumum operations
__kernel void CONVOLUTION_LAYER_2(const __global float * input_feature,
								  const __global float *  conv_kernel,
								  const __global float *  conv_bias,
								  __global float *  output_feature)
{
	// Connection Table for Dummy Operation
	/*
	3 Input feature map (6)
	----------------------------------------
	{ 1, 2, 3, 0, 0, 0 }, // 1,2 + 3 --> 2,3
	{ 0, 2, 3, 4, 0, 0 }, // 2,3 + 4 --> 3,4
	{ 0, 0, 3, 4, 5, 0 }, // 3,4 + 5 --> 4,5
	{ 0, 0, 0, 4, 5, 6 }, // 4,5 + 6 --> 5,6
	{ 1, 0, 0, 0, 5, 6 }, // 5,6 + 1 --> 6,1
	{ 1, 2, 0, 0, 0, 6 }, // 6,1 + 2

	4 Input feature map (9)
	----------------------------------------
	{ 1, 2, 3, 4, 0, 0 }, // 1,2,3 + 4
	{ 0, 2, 3, 4, 5, 0 }, // 2,3,4 + 5
	{ 0, 0, 3, 4, 5, 6 }, // 3,4,5 + 6
	{ 1, 0, 0, 4, 5, 6 }, // 4,5,6 + 1
	{ 1, 2, 0, 0, 5, 6 }, // 5,6,1 + 2
	{ 1, 2, 3, 0, 0, 6 }, // 6,1,2 + 3
	{ 1, 2, 0, 4, 5, 0 }, // 1,4 + 2,5
	{ 0, 2, 3, 0, 5, 6 }, // 2,5 + 3,6
	{ 1, 0, 3, 4, 0, 6 }, // 3,6 + 4,1

	6 Input feature map (1)
	----------------------------------------
	{ 1, 2, 3, 4, 5, 6 }  // 4,1 + 5,2
	*/

	int col, row;
	int row_f, col_f, row_x, col_x, depth_x;
	int depth_in, depth_out, batch_idx;
	int dst_temp;
	int src_idx, dst_idx, conv_idx;
	int src_temp, conv_temp;
	int row_index, col_index;
	float value;
	int i, j;

	// Private or Locacl Memory
	float buff_input_feature[25];
	float buff_conv_kernel[25];

	// Work-item index
	batch_idx = get_global_id(0) / CONV_2_TYPE;					// 0 ~ 10
	row = get_global_id(1) / CONV_2_OUTPUT_WH;		// 0 ~ 10
	col = get_global_id(1) % CONV_2_OUTPUT_WH;		// 0 ~ 10
	depth_out = get_global_id(0) % CONV_2_TYPE;					// 0 ~ 16

	// Feature Map Start Index
	src_idx = batch_idx * CONV_1_TYPE * CONV_2_INPUT_SIZE;
	dst_idx = batch_idx * CONV_2_OUTPUT_SIZE * CONV_2_TYPE;
	
	// Feature Map Start Sub-Index(depth_out)
	conv_idx = CONV_2_WH * CONV_2_WH * CONV_1_TYPE * depth_out;
	dst_temp = depth_out * CONV_2_OUTPUT_SIZE;

	// Initialization of reduction value
	value = 0;

	// Multiplication by Convolution kernel and Input feautre map
	for (depth_in = 0; depth_in < POOL_1_TYPE; depth_in++)
	{
		//Feature Map Start Sub-Index(depth_in)
		src_temp = depth_in * CONV_2_INPUT_SIZE + src_idx;
		conv_temp = depth_in * CONV_2_WH * CONV_2_WH + conv_idx;

		// Copy Input feature map to Private memory
		for (row_x = 0; row_x < CONV_2_WH; row_x++)
		{
			for (col_x = 0; col_x < CONV_2_WH; col_x++)
			{				
				buff_input_feature[row_x * 5 + col_x] = input_feature[src_temp + CONV_2_INPUT_WH * (row + row_x) + col + col_x];
				buff_conv_kernel[row_x * 5 + col_x] = conv_kernel[conv_temp + row_x * 5 + col_x];
				//value += buff_input_feature[row_x * 5 + col_x] * buff_conv_kernel[row_x * 5 + col_x];
			}
		}
		for (row_x = 0; row_x < CONV_2_WH; row_x++)
		{
			for (col_x = 0; col_x < CONV_2_WH; col_x++)
			{
				value += buff_input_feature[row_x * 5 + col_x] * buff_conv_kernel[row_x * 5 + col_x];
			}
		}
	}
	
	// Result of Convolution (Activation and Bias)
	//output_feature[dst_idx + dst_temp + CONV_2_OUTPUT_WH * row + col] =  tanh(value + conv_bias[depth_out]);
	output_feature[dst_idx + dst_temp + CONV_2_OUTPUT_WH * row + col] = value + conv_bias[depth_out];
}

// Images are organized like [Batch_Index][Depth_In_Index][Row_Index][Col_Index] 
// Filters are organized like [Depth_Out_Index][Depth_In_Index][Row_Index][Col_Index] 
// Output are organized like [Depth_Out_Index][Row_Index][Col_Index] 
// Work size is organized like [Image_Batch][POOL_2_OUTPUT_WH * POOL_2_OUTPUT_WH][CONV_2_TYPE] 
// Make more thread by minumum operations
__kernel void  POOLING_LAYER2(const __global float * input_feature,
							  const __global float *  pool_kernel,
							  const __global float *  pool_bias,
							  __global float *  output_feature)
{
	int col, row;
	int depth_out, batch_cnt;
	int src_idx, dst_idx;
	float value = 0;

	// Work-item index
	batch_cnt = get_global_id(0) / POOL_2_OUTPUT_WH;
	depth_out = get_global_id(1) / POOL_2_OUTPUT_WH;
	row = get_global_id(0) % POOL_2_OUTPUT_WH;
	col = get_global_id(1) % POOL_2_OUTPUT_WH;
	
	//for (batch_cnt = 0; batch_cnt < image_Batch; batch_cnt++) {

		// Feature Map Start Index
		src_idx = batch_cnt * POOL_2_INPUT_SIZE * POOL_2_TYPE + depth_out * POOL_2_INPUT_SIZE;
		dst_idx = batch_cnt * POOL_2_OUTPUT_SIZE * POOL_2_TYPE + depth_out * POOL_2_OUTPUT_SIZE;

		// Multiplication by Convolution kernel and Input feautre map
		value = input_feature[src_idx + (row * 2) * POOL_2_INPUT_WH + (col * 2)] * pool_kernel[depth_out * POOL_2_SIZE]
			+ input_feature[src_idx + (row * 2) * POOL_2_INPUT_WH + (col * 2 + 1)] * pool_kernel[depth_out * POOL_2_SIZE + 1]
			+ input_feature[src_idx + (row * 2 + 1) * POOL_2_INPUT_WH + (col * 2)] * pool_kernel[depth_out * POOL_2_SIZE + 2]
			+ input_feature[src_idx + (row * 2 + 1) * POOL_2_INPUT_WH + (col * 2 + 1)] * pool_kernel[depth_out * POOL_2_SIZE + 3];

		value *= 2.7;

		// Result of Pooling (Activation and Bias)
		//output_feature[dst_idx + row * POOL_2_OUTPUT_WH + col] = tanh(value + pool_bias[depth_out]);
		output_feature[dst_idx + row * POOL_2_OUTPUT_WH + col] = value + pool_bias[depth_out];


}

__kernel void CONVOLUTION_LAYER_3(const __global float * input_feature,
								  const __global float * conv_kernel,
								  const __global float * conv_bias,
								  __global float * output_feature)
{
	int col_f, row_f;
	int local_idx;

	int depth_in, batch_cnt, depth_out;
	int src_idx, dst_idx;
	int conv_idx, conv_idx_0;
	int src_temp;
	float value = 0;

	// Private or Locacl Memory
	float buff_input_feature[CONV_3_INPUT_SIZE];
	float buff_conv_kernel[CONV_3_INPUT_SIZE];

	// Work-item index
	batch_cnt = get_global_id(0);
	depth_out = get_global_id(1);

	// Feature Map Start Index
	src_idx = batch_cnt * POOL_2_TYPE * CONV_3_INPUT_SIZE;
	dst_idx = batch_cnt * CONV_3_TYPE;
	conv_idx_0 = depth_out * CONV_3_INPUT_SIZE * POOL_2_TYPE;

	// Initialization of reduction value
	value = 0;


	// Multiplication by Convolution and Input feature maps
	for (depth_in = 0; depth_in < POOL_2_TYPE; depth_in++)
	{
		src_temp = depth_in * CONV_3_INPUT_SIZE;
		conv_idx = depth_in * CONV_3_INPUT_SIZE + conv_idx_0;

		// Copy Kernel to Private memory
		for (row_x = 0; row_x < 5; row_x++)
		{
			//for (col_x = 0; col_x < 5; col_x++)
			{
				/*
				buff_conv_kernel[row_x * 5 + col_x] = conv_kernel[conv_idx + CONV_3_WH * row_x + col_x];
				buff_input_feature[row_x * 5 + col_x] = input_feature[src_idx + src_temp + row_x * 5 + col_x];
				value += buff_input_feature[CONV_3_WH * row_f + col_f] * buff_conv_kernel[CONV_3_WH * row_f + col_f];
			*/
				buff_conv_kernel[row_x * 5 ] = conv_kernel[conv_idx + CONV_3_WH * row_x];
				buff_input_feature[row_x * 5] = input_feature[src_idx + src_temp + row_x * 5];

				buff_conv_kernel[row_x * 5 + 1] = conv_kernel[conv_idx + CONV_3_WH * row_x + 1];
				buff_input_feature[row_x * 5 + 1] = input_feature[src_idx + src_temp + row_x * 5 + 1];

				buff_conv_kernel[row_x * 5 + 2] = conv_kernel[conv_idx + CONV_3_WH * row_x + 2];
				buff_input_feature[row_x * 5 + 2] = input_feature[src_idx + src_temp + row_x * 5 + 2];

				buff_conv_kernel[row_x * 5 + 3] = conv_kernel[conv_idx + CONV_3_WH * row_x + 3];
				buff_input_feature[row_x * 5 + 3] = input_feature[src_idx + src_temp + row_x * 5 + 3];

				buff_conv_kernel[row_x * 5 + 4] = conv_kernel[conv_idx + CONV_3_WH * row_x + 4];
				buff_input_feature[row_x * 5 + 4] = input_feature[src_idx + src_temp + row_x * 5 + 4];

				value += buff_input_feature[CONV_3_WH * row_x] * buff_conv_kernel[CONV_3_WH * row_x]
					+ buff_input_feature[CONV_3_WH * row_x + 1] * buff_conv_kernel[CONV_3_WH * row_x + 1]
					+ buff_input_feature[CONV_3_WH * row_x + 2] * buff_conv_kernel[CONV_3_WH * row_x + 2]
					+ buff_input_feature[CONV_3_WH * row_x + 3] * buff_conv_kernel[CONV_3_WH * row_x + 3]
					+ buff_input_feature[CONV_3_WH * row_x + 4] * buff_conv_kernel[CONV_3_WH * row_x + 4];

			}
		}
	}

	// Result of Convolution (Activation and Bias)
	//output_feature[dst_idx + depth_out] = tanh(value + conv_bias[depth_out]);
	output_feature[dst_idx + depth_out] = 1;//(value + conv_bias[depth_out]);
}


__kernel void FULLY_CONNECTED_LAYER_1(const __global  * Input_NN,
	const __global  float *  Weight,
	const __global  float *  Bias,
	__global float *  Output_NN)
{
	int feed_forward_i, feed_forward_j, batch_cnt;

	batch_cnt = get_global_id(0);
	feed_forward_j = get_global_id(1);

	int src_idx, dst_idx;

	//for (batch_cnt = 0; batch_cnt<image_Batch; batch_cnt++)
	{
		src_idx = batch_cnt * INPUT_NN_1_SIZE;
		dst_idx = batch_cnt * INPUT_NN_2_SIZE;

		//for (feed_forward_j = 0; feed_forward_j < INPUT_NN_2_SIZE; feed_forward_j++)
		{
			// Init
			Output_NN[batch_cnt * INPUT_NN_2_SIZE + feed_forward_j] = 0;

			// Multiplication by Input node and Weight(ÃßÈÄ PRIVATE MEMORY)
			for (feed_forward_i = 0; feed_forward_i < INPUT_NN_1_SIZE; feed_forward_i++)
			{
				Output_NN[dst_idx + feed_forward_j] += Input_NN[src_idx + feed_forward_i] * Weight[feed_forward_i * INPUT_NN_2_SIZE + feed_forward_j];
			}

			// Result of FC
			Output_NN[dst_idx + feed_forward_j] = tanh(Output_NN[dst_idx + feed_forward_j] + Bias[feed_forward_j]);
		}
	}
}


__kernel void FULLY_CONNECTED_LAYER_2(const __global  * Input_NN,
	const __global  float *  Weight,
	const __global  float *  Bias,
	__global float *  Output_NN)
{
	int feed_forward_i, feed_forward_j, batch_cnt;

	batch_cnt = get_global_id(0);
	feed_forward_j = get_global_id(1);

	int src_idx, dst_idx;

	//for (batch_cnt = 0; batch_cnt<image_Batch; batch_cnt++)
	{
		src_idx = batch_cnt * INPUT_NN_2_SIZE;
		dst_idx = batch_cnt * OUTPUT_NN_2_SIZE;

		//for (feed_forward_j = 0; feed_forward_j < OUTPUT_NN_2_SIZE; feed_forward_j++)
		{
			// Init
			Output_NN[batch_cnt * OUTPUT_NN_2_SIZE + feed_forward_j] = 0;

			// Multiplication by Input node and Weight
			for (feed_forward_i = 0; feed_forward_i < INPUT_NN_2_SIZE; feed_forward_i++)
			{
				Output_NN[dst_idx + feed_forward_j] += Input_NN[src_idx + feed_forward_i] * Weight[feed_forward_i * OUTPUT_NN_2_SIZE + feed_forward_j];
			}

			// Result of FC
			Output_NN[dst_idx + feed_forward_j] = tanh(Output_NN[dst_idx + feed_forward_j] + Bias[feed_forward_j]);
		}
	}
}

/*
// Time is 27~30ms
// OpenMP Version is 7~8ms haha....
__kernel void CONVOLUTION_LAYER_1(__global float * input_feature,
__global float * conv_kernel,
__global float * conv_bias,
__global float * output_feature)
{
// Time is 21 ~25 ms , 389ms

int batch = get_global_id(1) / 28 / 5;
int out = get_global_id(0) / 28 / 5;

int col_idx = get_global_id(0) % 28 % 5;
int row_idx = get_global_id(1) % 28 % 5;

float sum=0;
int row_f = get_local_id(0);
int col_f = get_local_id(1);

//for(unsigned int in=0; in <1; in++)
{
__local float buff_input_feature[25];
__local float buff_conv_kernel[25];

//Copy data to loal memory
//for (int row_f = 0; row_f < 5; row_f++)
{
//for (int col_f = 0; col_f < 5; col_f++)
{
buff_input_feature[row_f * 5 + col_f] = input_feature[batch * 32 * 32 +
(row_idx + row_f) * 32 + col_idx + col_f];
buff_conv_kernel[row_f * 5 + col_f] = conv_kernel[out * 25 + row_f * 5 + col_f];
}
}
barrier(CLK_LOCAL_MEM_FENCE);

//Compute the convolution
//for(int i=0; i<25; i=i+5)
{
sum += buff_input_feature[row_f * 5 + col_f] * buff_conv_kernel[row_f * 5 + col_f];

}
}

int out_idx = batch * 28 * 28 * 6 + out * 28 * 28 + row_idx * 28 + col_idx;
output_feature[out_idx] = tanh(sum + conv_bias[out]);

}
*/