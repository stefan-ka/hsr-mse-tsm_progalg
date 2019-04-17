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
			uint4 pixel = read_imageui(source, sampler, coords);
			int f = hFilter[filterIdx];

			hC.x += f*pixel.x;
			hC.y += f*pixel.y;
			hC.z += f*pixel.z;

			f = vFilter[filterIdx];
			vC.x += f*pixel.x;
			vC.y += f*pixel.y;
			vC.z += f*pixel.z;

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

////////////////////////////////////////////////////////////////////////
// OpenCL kernel
/*
__kernel void edgesWithLocalMem(__read_only image2d_t source, __write_only image2d_t dest, __constant int* hFilter, __constant int* vFilter, int fSize, sampler_t sampler, __local uint4* localImage) {
	// TODO implement edge detection with copying the part of the global input image accessed by the workgroup to local memory.
	// use read_imageui(...) and write_imageui(...) to read/write one pixel of source/dest
	// 1st step: each work item fills in one or two rows of the local memory and waits at a barrier
	// 2nd step: as soon as the last work item reaches the barrier the local memory is complete
	// 3rd step: run edge detection similar to the implementation without local memory
	const int w = get_global_size(0);
	const int h = get_global_size(1);
	const int lcol = get_local_id(0);
	const int lrow = get_local_id(1);
	const int lnCols = get_local_size(0);
	const int lnRows = get_local_size(1);
	const int fSizeD2 = fSize/2;
	const int startCol = get_group_id(0)*lnCols - fSizeD2; // group-id is the index of the working group, local_size is the tile size
	const int startRow = get_group_id(1)*lnRows - fSizeD2; // group-id is the index of the working group, local_size is the tile size
	const int lw = lnCols + 2*fSizeD2;
	const int lh = lnRows + 2*fSizeD2;

	int2 coords;

	for (int i=lrow; i < lh; i += lnRows) {
		coords.y = startRow + i;
		int offset = i*lw;
		for (int j=lcol; j < lw; j += lnCols) {
			coords.x = startCol + j;
			uint4 pixel = read_imageui(source, sampler, coords);
			localImage[offset + j] = pixel; // compiler hangs here
			//localImage[offset + j] = read_imageui(source, sampler, coords);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	const int col = get_global_id(0);
	const int row = get_global_id(1);
	int4 hC = { 0, 0, 0, 0 };
	int4 vC = { 0, 0, 0, 0 };
	int filterIdx = 0;
	
	int offset = lrow*lw;
	for(int i = lrow; i < lrow + fSize; i++) {
		for(int j = lcol; j < lcol + fSize; j++) {
			uint4 pixel = localImage[offset + j];
			int f = hFilter[filterIdx];

			hC.x += f*pixel.x;
			hC.y += f*pixel.y;
			hC.z += f*pixel.z;
			f= vFilter[filterIdx];
			vC.x += f*pixel.x;
			vC.y += f*pixel.y;
			vC.z += f*pixel.z;

			filterIdx++;
		}
		offset += lw;
	}

	if (row < h && col < w) {
		// pixel lies inside output image
		coords.x = col;
		coords.y = row;
		uint4 p = { dist(hC.x, vC.x), dist(hC.y, vC.y), dist(hC.z, vC.z), 255 };

		write_imageui(dest, coords, p);
	}
}
*/