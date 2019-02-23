#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>
#include "Stopwatch.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
// Prototypes
void matMultSeq(const int* a, const int* b, int* const c, const int n);
void cannonBlocking(int* const a, int* const b, int* const c, const int n, const int pSqrt);
void cannonNonBlocking(int* const a, int* const b, int* const c, const int n, const int pSqrt);

//////////////////////////////////////////////////////////////////////////////////////////////
static bool different(const int* const c1, const int* const c2, int n2) {
	return memcmp(c1, c2, sizeof(int)*n2) != 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Matrix multiplaction tests
int main() {
	Stopwatch swCPU, swMPI;
	int wrongMPIresults = 0;
	int nProcs, myID;
	bool nonBlocking = false;

	MPI_Init(nullptr, nullptr);
	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myID);

	// set up the Cartesian topology with wrapparound connections and rank reordering
	int pSqrt = (int)sqrt(nProcs);

	if (pSqrt*pSqrt != nProcs) {
		if (myID == 0) cerr << "number of processes " << nProcs << " must be a square number" << endl;
		MPI_Finalize(); 
		return -1;
	} else {
		if (myID == 0) {
			cout << "Non-blocking [true/false] ";
			cin >> boolalpha >> nonBlocking;
		}
		// send nonBlocking to all processes 
		MPI_Bcast(&nonBlocking, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

		if (nonBlocking) {
			cout << "Cannon's non-blocking algorithm started" << endl;
		} else {
			cout << "Cannon's blocking algorithm started" << endl;
		}
	}

	for (int n = 1000; n <= 2000; n += 200) {
		const int nLocal = n/pSqrt;
		const int nLocal2 = nLocal*nLocal;
		const int maxVal = (int)sqrt(INT_MAX/n);

		int *aLocal = new int[nLocal2];
		int *bLocal = new int[nLocal2];
		int *cLocal = new int[nLocal2];

		if (myID == 0) {
			const int n1 = nLocal*pSqrt;
			const int n2 = n1*n1;

			int *a = new int[n2];
			int *b = new int[n2];
			int *c0 = new int[n2];
			int *c1 = new int[n2];
			int *tmp = new int[n2];

			for (int i = 0; i < n2; i++) {
				a[i] = maxVal*rand()/RAND_MAX;
				b[i] = maxVal*rand()/RAND_MAX;
				//a[i] = (i < n) ? 1 : 0;
				//b[i] = (i%n == 0) ? 1 : 0;
			}

			// run serial implementation
			swCPU.Restart();
			memset(c0, 0, n2*sizeof(int));
			matMultSeq(a, b, c0, n1);
			swCPU.Stop();

			//int x; cout << "enter x: "; cin >> x;

			// partition and distribute matrix A
			const int size = nLocal*sizeof(int);
			int *t = tmp;

			for (int i = 0; i < pSqrt; i++) {
				for (int j = 0; j < pSqrt; j++) {
					int *arow = a + i*nLocal*n1 + j*nLocal;

					// copy block a(ij) to array tmp
					for (int k = 0; k < nLocal; k++) {
						memcpy(t, arow, size);
						t += nLocal;
						arow += n1;
					}
				}
			}
			MPI_Scatter(tmp, nLocal2, MPI_FLOAT, aLocal, nLocal2, MPI_FLOAT, 0, MPI_COMM_WORLD);

			// partition and distribute matrix B
			t = tmp;
			for (int i = 0; i < pSqrt; i++) {
				for (int j = 0; j < pSqrt; j++) {
					int *brow = b + i*nLocal*n1 + j*nLocal;

					// copy block b(ij) to array tmp
					for (int k = 0; k < nLocal; k++) {
						memcpy(t, brow, size);
						t += nLocal;
						brow += n1;
					}
				}
			}
			MPI_Scatter(tmp, nLocal2, MPI_FLOAT, bLocal, nLocal2, MPI_FLOAT, 0, MPI_COMM_WORLD);

			// run MPI matrix multiplication
			swMPI.Restart();
			if (nonBlocking) cannonNonBlocking(aLocal, bLocal, cLocal, nLocal, pSqrt);
			else cannonBlocking(aLocal, bLocal, cLocal, nLocal, pSqrt);
			//cout << swMPI.GetSplitTimeMilliseconds() << endl;
			swMPI.Stop();

			// gather matrix C
			MPI_Gather(cLocal, nLocal2, MPI_FLOAT, tmp, nLocal2, MPI_FLOAT, 0, MPI_COMM_WORLD);

			t = tmp;
			for (int i = 0; i < pSqrt; i++) {
				for (int j = 0; j < pSqrt; j++) {
					int *crow = c1 + i*nLocal*n1 + j*nLocal;

					// copy array tmp to block c1(ij)
					for (int k = 0; k < nLocal; k++) {
						memcpy(crow, t, size);
						t += nLocal;
						crow += n1;
					}
				}
			}

			if (different(c0, c1, n2) && !wrongMPIresults) wrongMPIresults = n1;

			// clean-up
			delete[] a;
			delete[] b;
			delete[] c0;
			delete[] c1;

		} else {
			MPI_Scatter(nullptr, nLocal2, MPI_FLOAT, aLocal, nLocal2, MPI_FLOAT, 0, MPI_COMM_WORLD);
			MPI_Scatter(nullptr, nLocal2, MPI_FLOAT, bLocal, nLocal2, MPI_FLOAT, 0, MPI_COMM_WORLD);

			// run MPI matrix multiplication
			if (nonBlocking) cannonNonBlocking(aLocal, bLocal, cLocal, nLocal, pSqrt);
			else cannonBlocking(aLocal, bLocal, cLocal, nLocal, pSqrt);

			MPI_Gather(cLocal, nLocal2, MPI_FLOAT, nullptr, nLocal2, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}

		// clean-up
		delete[] aLocal;
		delete[] bLocal;
		delete[] cLocal;
	}

	if (myID == 0) {
		const double cpuTime = swCPU.GetElapsedTimeMilliseconds();
		cout << "CPU wall-clock time = " << cpuTime << " ms" << endl;
		if (wrongMPIresults) {
			cout << "MPI results are invalid (matrix size " << wrongMPIresults << ")" << endl;
		} else {
			const double mpiTime = swMPI.GetElapsedTimeMilliseconds();
			const double speedup = cpuTime/mpiTime;
			cout << "MPI results are valid, wall-clock time = " << mpiTime << " ms, S = " << speedup << ", E = " << speedup/nProcs << endl;
		}
	}

	MPI_Finalize();
}