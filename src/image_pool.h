#include "common.h"

void POOLING_LAYER_1(float src[POOL_1_TYPE * image_Batch][POOL_1_INPUT_WH * POOL_1_INPUT_WH], 
					float pool_kernel[POOL_1_TYPE][POOL_1_SIZE],
					float pool_bias[POOL_1_TYPE],
					float dst[POOL_1_TYPE * image_Batch][POOL_1_OUTPUT_WH * POOL_1_OUTPUT_WH],
					int scale_factor)
{
	int row, col, row_sub, col_sub, batch_cnt, depth;
	float value;

	for (batch_cnt = 0; batch_cnt < image_Batch; batch_cnt++)
	{
		for (depth = 0; depth < POOL_1_TYPE; depth++)
		{
			for (row = 0; row < POOL_1_OUTPUT_WH; row++)
			{
				for (col = 0; col < POOL_1_OUTPUT_WH; col++)
				{
					// Computation of Pooling
					value = src[depth + POOL_1_TYPE * batch_cnt][(row * 2) * POOL_1_INPUT_WH + (col * 2)] * pool_kernel[depth][0]
						+ src[depth + POOL_1_TYPE * batch_cnt][(row * 2) * POOL_1_INPUT_WH + (col * 2 + 1)] * pool_kernel[depth][1]
						+ src[depth + POOL_1_TYPE * batch_cnt][(row * 2 + 1) * POOL_1_INPUT_WH + (col * 2)] * pool_kernel[depth][2]
						+ src[depth + POOL_1_TYPE * batch_cnt][(row * 2 + 1) * POOL_1_INPUT_WH + (col * 2 + 1)] * pool_kernel[depth][3];
			
					value *= 2.7;

					// Activation function
					dst[batch_cnt * POOL_1_TYPE + depth][row * POOL_1_OUTPUT_WH + col] = tanh(value + pool_bias[depth]);
				}
			}
		}
	}
}


void POOLING_LAYER_2(float src[POOL_2_TYPE * image_Batch][POOL_2_INPUT_WH * POOL_2_INPUT_WH],
					 float pool_kernel[POOL_2_TYPE][POOL_2_SIZE],
					 float pool_bias[POOL_2_TYPE],
					 float dst[POOL_2_TYPE * image_Batch][POOL_2_OUTPUT_WH * POOL_2_OUTPUT_WH],
					 int scale_factor)
{
	int row, col, row_sub, col_sub, batch_cnt, depth;
	float value;

	for (batch_cnt = 0; batch_cnt < image_Batch; batch_cnt++)
	{
		for (depth = 0; depth < POOL_2_TYPE; depth++)
		{
			for (row = 0; row < POOL_2_OUTPUT_WH; row++)
			{
				for (col = 0; col < POOL_2_OUTPUT_WH; col++)
				{
					// Computation of Pooling
					value = src[depth + POOL_2_TYPE * batch_cnt][(row * 2) * POOL_2_INPUT_WH + (col * 2)] * pool_kernel[depth][0]
						+ src[depth + POOL_2_TYPE * batch_cnt][(row * 2) * POOL_2_INPUT_WH + (col * 2 + 1)] * pool_kernel[depth][1]
						+ src[depth + POOL_2_TYPE * batch_cnt][(row * 2 + 1) * POOL_2_INPUT_WH + (col * 2)] * pool_kernel[depth][2]
						+ src[depth + POOL_2_TYPE * batch_cnt][(row * 2 + 1) * POOL_2_INPUT_WH + (col * 2 + 1)] * pool_kernel[depth][3];
			
					value *= 2.7;

					// Activation function
					dst[batch_cnt * POOL_2_TYPE + depth][row * POOL_2_OUTPUT_WH + col] = tanh(value + pool_bias[depth]);
				}
			}
		}
	}
}
