#include <mpi.h> 
#include <iostream>
#include <cassert>
#include <algorithm>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////
// compare-split of nlocal data elements
// pre-condition: sent contains the data sent to another process
//                received contains the data received by the other process
// post-condition: result contains the kept data elements
//                changed is true if result contains received data
static void compareSplit(int nlocal, float *sent, float *received, float *result, bool keepSmall, bool& changed) {
	changed = false;
	if (keepSmall) {
		for (int i = 0, j = 0, k = 0; k < nlocal; k++) {
			if (j == nlocal || (i < nlocal && sent[i] <= received[j])) {
				result[k] = sent[i++];
			} else {
				result[k] = received[j++];
				changed = true;
			}
		}
	} else {
		const int last = nlocal - 1;
		for (int i = last, j = last, k = last; k >= 0; k--) {
			if (j == -1 || (i >= 0 && sent[i] >= received[j])) {
				result[k] = sent[i--];
			} else {
				result[k] = received[j--];
				changed = true;
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// elements to be sorted
// received and temp are auxiliary buffers
static void oddEvenSort(const int nProcs, const int nlocal, const int myID, float *elements, float *received, float *temp) {
	int oddid, evenid;
	MPI_Status status;
	bool changed = true;

	// determine the id of the processors that myID needs to communicate during the odd and even phases
	if (myID & 1) {
		oddid = myID + 1;
		evenid = myID - 1;
	} else {
		oddid = myID - 1;
		evenid = myID + 1;
	}
	if (evenid < 0 || evenid == nProcs) evenid = MPI_PROC_NULL;
	if (oddid  < 0 || oddid  == nProcs) oddid = MPI_PROC_NULL;

	// main loop of odd-even sort: local data to send is in elements buffer
	for (int i = 0; i < nProcs && changed; i++) {
		bool lChanged = false;

		if (i & 1) {
			// odd phase
			MPI_Sendrecv(elements, nlocal, MPI_FLOAT, oddid, 1, received, nlocal, MPI_FLOAT, oddid, 1, MPI_COMM_WORLD, &status);
		} else {
			// even phase
			MPI_Sendrecv(elements, nlocal, MPI_FLOAT, evenid, 1, received, nlocal, MPI_FLOAT, evenid, 1, MPI_COMM_WORLD, &status);
		}
		if (status.MPI_SOURCE != MPI_PROC_NULL) {
			// sent data in elements buffer
			// received data in received buffer
			compareSplit(nlocal, elements, received, temp, myID < status.MPI_SOURCE, lChanged);
			// temp contains result of compare-split operation: copy temp back to elements buffer
			memcpy(elements, temp, nlocal*sizeof(float));
		}

		// reduce changed value: all processes have the same information in changed
		MPI_Allreduce(&lChanged, &changed, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
	}

}

//////////////////////////////////////////////////////////////////////////////////////////////////
// nProcs: number of processes
// nlocal: number of elements to be sorted
// elements: data to be sorted
void shellSort(const int nProcs, const int nlocal, const int myID, float *elements) {
	// TODO
}

//////////////////////////////////////////////////////////////////////////////////////////////////
void shellsort() {
	int nProcs, myID;
	double seqElapsed = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &myID);
	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
	if (myID == 0) {
		cout << "Shellsort with " << nProcs << " MPI processes" << endl;
	}

	for (int i = 15; i <= 27; i += 3) {
		const int n = 1 << i;
		const int nlocal = n/nProcs;
		assert(nlocal*nProcs == n);

		float *elements = nullptr;
		float *received = new float[nlocal];

		if (myID == 0) {
			// fill in elements with random numbers
			elements = new float[n];
			for (int i = 0; i < n; i++) {
				elements[i] = (float)rand();
			}

			// make copy and sort the copy with std::sort
			float *copy = new float[n];
			memcpy(copy, elements, n*sizeof(float));
			double seqStart = MPI_Wtime();
			sort(copy, copy + n);
			seqElapsed = MPI_Wtime() - seqStart;
			delete[] copy;

		} else {
			elements = new float[nlocal];
		}

		// send partitioned elements to processes
		MPI_Scatter(elements, nlocal, MPI_FLOAT, received, nlocal, MPI_FLOAT, 0, MPI_COMM_WORLD);

		// use a barrier to synchronize start time
		MPI_Barrier(MPI_COMM_WORLD);
		double start = MPI_Wtime();
		shellSort(nProcs, nlocal, myID, received);

		// stop time
		double localElapsed = MPI_Wtime() - start, elapsed;

		// reduce maximum time
		MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		// send sorted local elements to process 0
		MPI_Gather(received, nlocal, MPI_FLOAT, elements, nlocal, MPI_FLOAT, 0, MPI_COMM_WORLD);

		// check if all elements are sorted in ascending order
		if (myID == 0) {
			int i = 1;
			while (i < nlocal && elements[i-1] <= elements[i]) i++;

			if (i == nlocal) {
				cout << n << " elements have been sorted in ascending order in " << elapsed << " seconds, speedup = " << seqElapsed/elapsed << endl;
			} else {
				cout << "elements are not correctly sorted" << endl;
			}
		}

		delete[] elements;
		delete[] received;
	}
}