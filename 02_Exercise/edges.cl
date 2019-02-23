////////////////////////////////////////////////////////////////////////
uint dist(int x, int y) {
	uint d = (uint)sqrt((float)(x*x) + (float)(y*y));
	return (d < 256) ? d : 255;
}

////////////////////////////////////////////////////////////////////////
// OpenCL kernel
__kernel void edges(__read_only image2d_t source, __write_only image2d_t dest, __constant int* hFilter, __constant int* vFilter, int fSize, sampler_t sampler) {
	const int w = get_global_size(0);
	const int h = get_global_size(1);

	// TODO implement edge detection without using local memory
	// use read_imageui(...) and write_imageui(...) to read/write one pixel of source/dest
}

