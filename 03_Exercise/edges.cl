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
	const int col = get_global_id(0);
	const int row = get_global_id(1);
	const int fSizeD2 = fSize/2;

	int4 hC = { 0, 0, 0, 0 };
	int4 vC = { 0, 0, 0, 0 };
	int filterIdx = 0;
	int2 coords;
	
	for(int i = -fSizeD2; i <= fSizeD2; i++) {
		coords.y = row + i;
		for(int j = -fSizeD2; j <= fSizeD2; j++) {
			coords.x = col + j;
			int4 pixel = convert_int4(read_imageui(source, sampler, coords));
			int hf = hFilter[filterIdx];

			int4 hFilterVector = { hf, hf, hf, 1 };
			hC = mad24(hFilterVector, pixel, hC);

			int vf = vFilter[filterIdx];
			int4 vFilterVector = { vf, vf, vf, 1 };
			vC = mad24(vFilterVector, pixel, vC);

			filterIdx++;
		}
	}

	if (row < h && col < w) {
		// pixel lies inside output image
		coords.x = col;
		coords.y = row;
		uint4 p = { dist(hC.x, vC.x), dist(hC.y, vC.y), dist(hC.z, vC.z), 255 };

		write_imageui(dest, coords, p);
	}
}

