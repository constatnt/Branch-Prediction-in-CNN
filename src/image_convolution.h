#include "common.cuh"

void kernel_load(char *fileName, float * target)
{
	int read_i = 0;
	FILE* read_ptr;

	if (read_ptr = fopen(fileName, "rb"))
	{
		while (EOF != fscanf(read_ptr, "%f", &target[read_i]))
		{
			read_i++;
		}
	}
	fclose(read_ptr);
	
}

// Convolution Layer 1
// Function by Batch_size(10)
// Input_feature_map[32x32],  Conv_kernel[6][25], Bias[6], Output_feature_map[6][28x28]
void CONVOLUTION_LAYER_1(float input_feature[image_Batch][INPUT_WH *INPUT_WH],
						float conv_kernel[CONV_1_TYPE][CONV_1_WH * CONV_1_WH],
						float conv_bias[CONV_1_TYPE],
						float output_feature[CONV_1_TYPE * image_Batch][CONV_1_OUTPUT_WH * CONV_1_OUTPUT_WH])
{
	int col, row, col_f, row_f;
	int depth_out, batch_cnt;

	float temp=0;
	
	for(batch_cnt=0; batch_cnt<image_Batch; batch_cnt++) {
		for (depth_out = 0; depth_out < CONV_1_TYPE; depth_out++) {
			for (row = 0; row < CONV_1_OUTPUT_WH; row++) {
				for (col = 0; col < CONV_1_OUTPUT_WH; col++) {

					// Init
					temp = 0;
								
					if (col == 6 && row == 8)
					{
						printf("Input image is \n");
						for (row_f = 0; row_f < CONV_1_WH; row_f++) {
							for (col_f = 0; col_f < CONV_1_WH; col_f++) {
								printf("%.10f  ", input_feature[batch_cnt][INPUT_WH * (row + row_f) + col + col_f]);
								
							}
							printf("\n");
						}
						printf("\n");
						printf("Convolution kernel is \n");
						for (row_f = 0; row_f < CONV_1_WH; row_f++) {
							for (col_f = 0; col_f < CONV_1_WH; col_f++) {
								printf("%.10f  ", conv_kernel[depth_out][CONV_1_WH * row_f + col_f]);
							}
							printf("\n");
						}
						printf("\n");
					}
					

					// Multiplication by Convolution and Input feature map
					for (row_f = 0; row_f < CONV_1_WH; row_f++) {
						for (col_f = 0; col_f < CONV_1_WH; col_f++) {
							temp += input_feature[batch_cnt][INPUT_WH * (row + row_f) + col + col_f] * conv_kernel[depth_out][CONV_1_WH * row_f + col_f];
						}
					}
					// Result of Convolution 
					output_feature[batch_cnt * CONV_1_TYPE + depth_out][CONV_1_OUTPUT_WH * row + col] = tanhf(temp + conv_bias[depth_out]);
				}
			}
		}
	}
}

// Convolution Layer 2
// Function by Batch_size(10)
// Input_feature_map[6][14x14],  Conv_kernel[16][6][25], Bias[16], Output_feature_map[16][10x10]
void CONVOLUTION_LAYER_2(float input_feature[CONV_1_TYPE * image_Batch][CONV_2_INPUT_WH *CONV_2_INPUT_WH],
	float conv_kernel[CONV_2_TYPE][CONV_1_TYPE][CONV_2_WH * CONV_2_WH],
	float conv_bias[CONV_2_TYPE],
	float output_feature[CONV_2_TYPE * image_Batch][CONV_2_OUTPUT_WH * CONV_2_OUTPUT_WH])
{
	// Connection Table for Dummy Operation
	/*
	3 Input feature map (6)
	----------------------------------------
	{ 1, 2, 3, 0, 0, 0 }, // 1,2 + 3 --> 2,3
	{ 0, 2, 3, 4, 0, 0 }, // 2,3 + 4 --> 3,4
	{ 0, 0, 3, 4, 5, 0 }, // 3,4 + 5 --> 4,5 V
	{ 0, 0, 0, 4, 5, 6 }, // 4,5 + 6 --> 5,6 V
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
	int col_f, row_f;
	int depth_in, depth_out;
	float temp = 0;
	int batch_idx;

	#ifdef omp_on
		#pragma omp parallel for private(batch_idx, depth_out, depth_in, row, col, row_f, col_f) shared(input_feature, conv_kernel, conv_bias, output_feature) reduction(+:temp) schedule(static)
	#endif

	for (batch_idx = 0; batch_idx < image_Batch; batch_idx++) {
		for (depth_out = 0; depth_out < CONV_2_TYPE; depth_out++) {
			for (row = 0; row < CONV_2_OUTPUT_WH; row++) {
				for (col = 0; col < CONV_2_OUTPUT_WH; col++) {

					// Init
					temp = 0;

					// Multiplication by Convolution and Input feature maps
					for (depth_in = 0; depth_in < CONV_1_TYPE; depth_in++) {
						for (row_f = 0; row_f < CONV_2_WH; row_f++) {
							for (col_f = 0; col_f < CONV_2_WH; col_f++) {
								temp += input_feature[batch_idx * CONV_1_TYPE + depth_in][CONV_2_INPUT_WH * (row_f + row) + col + col_f] * conv_kernel[depth_out][depth_in][CONV_2_WH * row_f + col_f];
							}
						}
					}
					// Result of Convolution
					output_feature[batch_idx * CONV_2_TYPE + depth_out][CONV_2_OUTPUT_WH * row + col] = tanhf(temp + conv_bias[depth_out]);
				}
			}
		}
	}

}

// Convolution Layer 3 (FC)
// Function by Batch_size(10)
// Input_feature_map[16][5x5],  Conv_kernel[120][16][5x5], Bias[120], Output_feature_map[120][1x1]
void CONVOLUTION_LAYER_3(float input_feature[CONV_2_TYPE*image_Batch][CONV_3_INPUT_WH *CONV_3_INPUT_WH],
						 float conv_kernel[CONV_3_TYPE][CONV_2_TYPE][CONV_3_WH * CONV_3_WH], float conv_bias[CONV_3_TYPE],
						 float output_feature[image_Batch * CONV_3_TYPE])
{
	int col, row, col_f, row_f;
	int depth_in, batch_cnt, depth_out;

	float temp=0;

	for (batch_cnt = 0; batch_cnt<image_Batch; batch_cnt++) {
		for (depth_out = 0; depth_out < CONV_3_TYPE; depth_out++) {

			// Init
			temp = 0;

			// Multiplication by Convolution and Input feature maps
			for (depth_in = 0; depth_in < POOL_2_TYPE; depth_in++) {
				for (row_f = 0; row_f < CONV_3_WH; row_f++) {
					for (col_f = 0; col_f < CONV_3_WH; col_f++) {
						temp += input_feature[POOL_2_TYPE * batch_cnt + depth_in][CONV_3_WH * row_f + col_f] * conv_kernel[depth_out][depth_in][CONV_3_WH * row_f + col_f];
					}
				}
			}
			// Result of Convolution
			output_feature[batch_cnt * CONV_3_TYPE + depth_out] = tanhf(temp+ conv_bias[depth_out]);
		}
	}
}

