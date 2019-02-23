#include <mpi.h>
#include <cmath>
#include <cassert>
#include <memory>

//////////////////////////////////////////////////////////////////////////////////////////////
// Cache aware serial implementation
// Matrix C has to be initialized with 0 in advance
void matMultSeq(const int* a, const int* b, int* const c, const int n) {
	int *crow = c;

	for (int i = 0; i < n; i++) {
		int bpos = 0;
		//for (int j = 0; j < n; j++) crow[j] = 0;
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
// Cannon's algorithm using blocking send and receive operations and Cartesian grid
void cannonBlocking(int* const a, int* const b, int* const c, const int n, const int pSqrt) {
	// set up the Cartesian topology with wrapparound connections and rank reordering
	int dims[] = { pSqrt, pSqrt };	// [y,x]
	int periods[] = { true, true };
	MPI_Comm comm2D;

	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm2D);

	// get rank and coordinates with respect to the new topology
	int my2Drank;
	int myCoords[2];	// [y,x]

	MPI_Comm_rank(comm2D, &my2Drank);
	MPI_Cart_coords(comm2D, my2Drank, 2, myCoords);

	// initialize C
	const int size = n*n;
	memset(c, 0, size*sizeof(int));

	// perform the initial matrix alignment: first for A then for B
	int shiftSrc, shiftDst;

	MPI_Cart_shift(comm2D, 1, -myCoords[0], &shiftSrc, &shiftDst);
	MPI_Sendrecv_replace(a, size, MPI_FLOAT, shiftDst, 1, shiftSrc, 1, comm2D, MPI_STATUSES_IGNORE);
	MPI_Cart_shift(comm2D, 0, -myCoords[1], &shiftSrc, &shiftDst);
	MPI_Sendrecv_replace(b, size, MPI_FLOAT, shiftDst, 1, shiftSrc, 1, comm2D, MPI_STATUSES_IGNORE);

	// compute ranks of the left and up shifts
	int leftRank, rightRank, downRank, upRank;

	MPI_Cart_shift(comm2D, 1, -1, &rightRank, &leftRank);
	MPI_Cart_shift(comm2D, 0, -1, &downRank, &upRank);

	// main computation loop
	for (int i = 0; i < pSqrt; i++) {
		// matrix multiplication: cLocal += aLocal * bLocal
		matMultSeq(a, b, c, n);

		// shift A left by one
		MPI_Sendrecv_replace(a, size, MPI_FLOAT, leftRank, 1, rightRank, 1, comm2D, MPI_STATUSES_IGNORE);

		// shift B up by one
		MPI_Sendrecv_replace(b, size, MPI_FLOAT, upRank, 1, downRank, 1, comm2D, MPI_STATUSES_IGNORE);
	}

	// restore the original distribution of A and B 
	MPI_Cart_shift(comm2D, 1, myCoords[0], &shiftSrc, &shiftDst);
	MPI_Sendrecv_replace(a, size, MPI_FLOAT, shiftDst, 1, shiftSrc, 1, comm2D, MPI_STATUSES_IGNORE);
	MPI_Cart_shift(comm2D, 0, myCoords[1], &shiftSrc, &shiftDst);
	MPI_Sendrecv_replace(b, size, MPI_FLOAT, shiftDst, 1, shiftSrc, 1, comm2D, MPI_STATUSES_IGNORE);

	// free up communicator
	MPI_Comm_free(&comm2D);
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Cannon's algorithm using non-blocking send and receive operations and Cartesian grid
void cannonNonBlocking(int* const a, int* const b, int* const c, const int n, const int pSqrt) {
	// TODO
}

