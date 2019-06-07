///////////////////////////////////////////////////////////////////////////////
__kernel void matrixmult(const __global int *a, const __global int *b, __global int *c) {
	const int w = get_global_size(0);
	const int h = get_global_size(1);
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	int value = 0;
	for(int i = 0; i < w; i++) {
		int valA = a[y * w + i];
		int valB = b[i * w + x];
		value += valA * valB;
	}

	c[y * w + x] = value;
}

