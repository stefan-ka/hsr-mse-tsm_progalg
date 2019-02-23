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
	// TODO: your parallel CPU implementation
}
