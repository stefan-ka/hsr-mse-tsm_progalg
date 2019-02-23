#include <iostream>
#include <cmath>
#include <climits>
#include <omp.h>
#include "Stopwatch.h"
#include "ocl.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
// Prototypes
void matMultSeq(const int* a, const int* b, int* const c, const int n);
void matMultCPU(const int* a, const int* b, int* const c, const int n);
void matMultGPU(OCLData& ocl, const int* a, const int* b, int* const c, const int n);
OCLData initOCL(const char* kernelFileName, const char* kernelName);

//////////////////////////////////////////////////////////////////////////////////////////////
static bool different(const int* const c1, const int* const c2, int n2) {
	return memcmp(c1, c2, sizeof(int)*n2) != 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Matrix multiplaction tests
int main() {
	Stopwatch swBase, swCPU, swGPU;
	int wrongCPUresults = 0, wrongGPUresults = 0;
	OCLData ocl = initOCL("matrixmult.cl", "matrixmult");

	for (int n = 1000; n <= 2000; n += 200) {
		const int n2 = n*n;
		const int maxVal = (int)sqrt(INT_MAX/n);

		int *a = new int[n2];
		int *b = new int[n2];
		int *c0 = new int[n2];
		int *c1 = new int[n2];

		for (int i = 0; i < n2; i++) {
			a[i] = maxVal*rand()/RAND_MAX;
			b[i] = maxVal*rand()/RAND_MAX;
		}

		// run serial implementation
		swBase.Restart();
		matMultSeq(a, b, c0, n);
		swBase.Stop();

		// run CPU matrix multiplication
		swCPU.Restart();
		matMultCPU(a, b, c1, n);
		swCPU.Stop();
		if (different(c0, c1, n2) && !wrongCPUresults) wrongCPUresults = n;
		memset(c1, 0, n2*sizeof(int));

		// run GPU matrix multiplication
		swGPU.Restart();
		matMultGPU(ocl, a, b, c1, n);
		swGPU.Stop();
		if (different(c0, c1, n2) && !wrongGPUresults) wrongGPUresults = 0;

		// clean-up
		delete[] a;
		delete[] b;
		delete[] c0;
		delete[] c1;
	}

	const double seqTime = swBase.GetElapsedTimeMilliseconds();
	cout << "Serial wall-clock time = " << seqTime << " ms" << endl;
	if (wrongCPUresults) {
		cout << "CPU results are invalid (matrix size " << wrongCPUresults << ")" << endl;
	} else {
		const double cpuTime = swCPU.GetElapsedTimeMilliseconds();
		const double speedup = seqTime/cpuTime;
		cout << "CPU results are valid, wall-clock time = " << cpuTime << " ms, S = " << speedup << ", E = " << speedup/omp_get_num_procs() << endl;
	}
	if (wrongGPUresults) {
		cout << "GPU results are invalid (matrix size " << wrongGPUresults << ")" << endl;
	} else {
		const double gpuTime = swGPU.GetElapsedTimeMilliseconds();
		const double speedup = seqTime/gpuTime;
		cout << "GPU results are valid, wall-clock time = " << gpuTime << " ms, S = " << speedup << ", E = " << speedup/ocl.m_computeUnits << endl;
	}
}