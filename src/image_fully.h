#include "common.h"

// 120 - > 84
void fully_connected_0(float Input_NN[10][120], 
					   float Weight[120][84], float Bias[84],
					   float Output_NN[10][84])
{
	int feed_forward_i, feed_forward_j, batch_cnt;

	#ifdef omp_on
		#pragma omp parallel for private(batch_cnt, feed_forward_j, feed_forward_i) shared(Output_NN, Input_NN, Weight, Bias)
	#endif
	for(batch_cnt=0; batch_cnt<image_Batch; batch_cnt++)
	{
		for (feed_forward_j = 0; feed_forward_j < 84; feed_forward_j++)	
		{ 
			// Init
			Output_NN[batch_cnt][feed_forward_j] = 0;

			// Multiplication by Input node and Weight
			for (feed_forward_i = 0; feed_forward_i < 120; feed_forward_i++)	
			{
				Output_NN[batch_cnt][feed_forward_j] += Input_NN[batch_cnt][feed_forward_i] * Weight[feed_forward_i][feed_forward_j];
			}

			// Result of FC
			Output_NN[batch_cnt][feed_forward_j] = tanh(Output_NN[batch_cnt][feed_forward_j] + Bias[feed_forward_j]);
		}
	}
}

// 84 -> 10
void fully_connected_1(float Input_NN[10][84], 
					   float Weight[84][10], float Bias[10],
					   float Output_NN[10][10])
{
	int feed_forward_i, feed_forward_j, batch_cnt;

	#ifdef omp_on
		#pragma omp parallel for private(batch_cnt, feed_forward_j, feed_forward_i) shared(Output_NN, Input_NN, Weight, Bias) 
	#endif
	for (batch_cnt = 0; batch_cnt<image_Batch; batch_cnt++)
	{
		for (feed_forward_j = 0; feed_forward_j < 10; feed_forward_j++)
		{
			// Init
			Output_NN[batch_cnt][feed_forward_j] = 0;
			
			// Multiplication by Input node and Weight
			for (feed_forward_i = 0; feed_forward_i < 84; feed_forward_i++)
			{
				Output_NN[batch_cnt][feed_forward_j] += Input_NN[batch_cnt][feed_forward_i] * Weight[feed_forward_i][feed_forward_j];
			}

			// Result of FC
			Output_NN[batch_cnt][feed_forward_j] = tanh(Output_NN[batch_cnt][feed_forward_j] + Bias[feed_forward_j]);
		}
	}


}