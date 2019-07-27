#include "common.h"

void READ_MNIST(char *fileName, unsigned char * source)
{
	FILE * read_ptr = fopen(fileName, "rb");
	fread(source, sizeof(unsigned char), MNIST_SIZE * image_Batch, read_ptr);
	fclose(read_ptr);
	
}

void READ_MNIST_LABEl(char *fileName, unsigned char * source)
{
	FILE * read_ptr = fopen(fileName, "rb");
	fread(source, sizeof(unsigned char), image_Batch, read_ptr);
	fclose(read_ptr);
}

float IMAGE_INIT(unsigned char *src, float *dst, int offset)
{
	int y, x;
	int batch_cnt;
	int scale_max=1.0;
	int scale_min=-0;

	for (batch_cnt = 0; batch_cnt < image_Batch; batch_cnt++)
	{
		for (y = 0; y < 28; y++)
		{
			for (x = 0; x < 28; x++)
			{
				dst[batch_cnt * 32 * 32 + 32 * (y + 2) + x + 2] 
					= (float)src[offset * image_Batch * 28 * 28 + batch_cnt * 28 * 28 + 28 * y + x]/ 255 * (scale_max- scale_min) + scale_min;
			}
		}
	}

}