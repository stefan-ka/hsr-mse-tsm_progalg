#include <mpi.h> 
#include <iostream>
#include <sstream>
#include <cassert>
#include <algorithm>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////
// compare-split of nlocal data elements
// pre-condition: sent contains the data sent to another process
//                received contains the data received by the other process
// pos-condition: result contains the kept data elements
static void CompareSplit(int nlocal, int *sent, int *received, int *result, bool keepSmall) {
	if (keepSmall) {
		for (int i = 0, j = 0, k = 0; k < nlocal; k++) {
			if (j == nlocal || (i < nlocal && sent[i] <= received[j])) {
				result[k] = sent[i++];
			} else {
				result[k] = received[j++];
			}
		}
	} else {
		const int last = nlocal - 1;
		for (int i = last, j = last, k = last; k >= 0; k--) {
			if (j == -1 || (i >= 0 && sent[i] >= received[j])) {
				result[k] = sent[i--];
			} else {
				result[k] = received[j--];
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////
void oddEvenSort() {
	const int n = 16000000;

	int nproc, myid, oddid, evenid;
	MPI_Status status;
	double sequentialTime = 0;

	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	int nlocal = n/nproc;
	assert(nlocal*nproc == n);

	int *elements = new int[nlocal];
	int *received = new int[nlocal];
	int *temp = new int[nlocal];

	// fill in elements with random numbers on process 0
	if (myid == 0) {
		elements = new int[n];
		for (int i = 0; i < n; i++) {
			elements[i] = rand();
		}

		// measure sequential time
		int *copy = new int[n];
		memcpy(copy, elements, n*sizeof(int));
		double start = MPI_Wtime();
		sort(copy, copy + n);
		sequentialTime = MPI_Wtime() - start;
		delete[] copy;
	} else {
		elements = new int[nlocal];
	}
	
	// distribute elements with scatter
	MPI_Scatter(elements, nlocal, MPI_INT, received, nlocal, MPI_INT, 0, MPI_COMM_WORLD);

	// start time measuring
	double startTime = MPI_Wtime();

	// sort local elements
	sort(received, received + nlocal);

	// determine the id of the processors that myid needs to communicate during the odd and even phases
	if (myid & 1) {
		oddid = myid + 1;
		evenid = myid - 1;
	} else {
		oddid = myid - 1;
		evenid = myid + 1;
	}
	if (evenid < 0 || evenid == nproc) evenid = MPI_PROC_NULL;
	if (oddid  < 0 || oddid  == nproc) oddid  = MPI_PROC_NULL;

	// main loop of odd-even sort: local data to send is in received buffer
	for (int i = 0; i < nproc; i++) {
		if (i & 1) {
			// odd phase
			MPI_Sendrecv(received, nlocal, MPI_INT, oddid, 1, elements, nlocal, MPI_INT, oddid, 1, MPI_COMM_WORLD, &status);
		} else {
			// even phase
			MPI_Sendrecv(received, nlocal, MPI_INT, evenid, 1, elements, nlocal, MPI_INT, evenid, 1, MPI_COMM_WORLD, &status);
		}
		if (status.MPI_SOURCE != MPI_PROC_NULL) {
			// sent data in received buffer
			// received data in elements buffer
			CompareSplit(nlocal, received, elements, temp, myid < status.MPI_SOURCE);
			// temp contains result of compare-split operation: copy temp back to received buffer
			memcpy(received, temp, nlocal*sizeof(int));
		}
	}

	// stop time measuring
	double localWallClockTime = MPI_Wtime() - startTime;
	
	double globalWallClockTime;
	MPI_Reduce(&localWallClockTime, &globalWallClockTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (myid == 0) {
		cout << "Largest wall-clock time is " << globalWallClockTime << endl;
		cout << "Sequential wall-clock time is " << sequentialTime << endl;
	}

	// all sorted data to process 0
	MPI_Gather(received, nlocal, MPI_INT, elements, nlocal, MPI_INT, 0, MPI_COMM_WORLD);

	// check if elements are sorted on process 0
	if (myid == 0) {
		bool sorted = true;
		for (int i = 1; i < n; i++) {
			if (elements[i - 1] > elements[i])
				sorted = false;
		}
		if (sorted) {
			cout << n << " elements have been sorted in ascending order" << endl;
		} else {
			cout << "elements are not correctly sorted" << endl;
		}
	}

	delete[] elements;
	delete[] received;
	delete[] temp;
}
