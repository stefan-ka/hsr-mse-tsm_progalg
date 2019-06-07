//////////////////////////////////////////////////////////////////////////////////////////////
// serial implementation with caching 
void matMultSeq(const int* a, const int* b, int* const c, const int n) {
	int *crow = c;

	for (int i = 0; i < n; i++) {
		int bpos = 0;
		for (int j = 0; j < n; j++) crow[j] = 0;
		for (int k = 0; k < n; k++) {
			for (int j = 0; j < n; j++) {
				crow[j] += a[k]*b[bpos++];
			}
		}
		a += n;
		crow += n;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Your best CPU matrix multiplication algorithm
void matMultCPU(const int* a, const int* b, int* const c, const int n) {
	#pragma omp parallel for default(none)
	for (int i = 0; i < n; i++) {
		const int* aPos = a + i * n;
		int *crow = c + i * n;
		int bpos = 0;
		for (int j = 0; j < n; j++) crow[j] = 0;
		for (int k = 0; k < n; k++) {
			for (int j = 0; j < n; j++) {
				crow[j] += aPos[k] * b[bpos++];
			}
		}
	}

	/* first naive approach. works, but does not speed up in comparison to serial
	#pragma omp parallel for default(none)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			int bpos = j;
			crow[i * n + j] = 0;
			for (int k = 0; k < n; k++) {
				crow[i * n + j] += a[i * n + k] * b[bpos];
				bpos += n;
			}
		}
	}*/
}
