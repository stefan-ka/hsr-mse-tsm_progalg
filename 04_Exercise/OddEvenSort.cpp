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

	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	int nlocal = n/nproc;
	assert(nlocal*nproc == n);

	int *elements = new int[nlocal];
	int *received = new int[nlocal];
	int *temp = new int[nlocal];

	// fill in elements with random numbers
	srand(myid);
	for (int i = 0; i < nlocal; i++) {
		elements[i] = rand();
	}

	// start time measuring
	// TODO

	// sort local elements
	sort(elements, elements + nlocal);

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
			MPI_Sendrecv(elements, nlocal, MPI_INT, oddid, 1, received, nlocal, MPI_INT, oddid, 1, MPI_COMM_WORLD, &status);
		} else {
			// even phase
			MPI_Sendrecv(elements, nlocal, MPI_INT, evenid, 1, received, nlocal, MPI_INT, evenid, 1, MPI_COMM_WORLD, &status);
		}
		if (status.MPI_SOURCE != MPI_PROC_NULL) {
			// sent data in received buffer
			// received data in elements buffer
			CompareSplit(nlocal, received, elements, temp, myid < status.MPI_SOURCE);
			// temp contains result of compare-split operation: copy temp back to elements buffer
			memcpy(elements, temp, nlocal*sizeof(int));
		}
	}

	// stop time measuring
	// TODO

	// check if local elements are sorted in ascending order
	int i = 1;
	while (i < nlocal && elements[i-1] <= elements[i]) i++;
	temp[0] = elements[0]; temp[1] = elements[nlocal - 1]; temp[2] = (i == nlocal);

	// check if all elements are sorted in ascending order
	if (myid == 0) {
		bool isSorted = true;
		int last = elements[nlocal - 1];

		// receive results in ascending order
		cout << "Receive data blocks of " << nproc << " processes." << endl;
		for (int i = 1; i < nproc; i++) {
			MPI_Recv(temp, 3, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (temp[0] < last || !temp[2]) {
				isSorted = false;
			}
			last = temp[1];
		}
		if (isSorted) {
			cout << n << " elements have been sorted in ascending order" << endl;
		} else {
			cout << "elements are not correctly sorted" << endl;
		}
	} else {
		MPI_Send(temp, 3, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}

	delete[] elements;
	delete[] received;
	delete[] temp;
}
