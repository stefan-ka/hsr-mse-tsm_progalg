#include <mpi.h> 
#include <iostream>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////
void oddEvenSort();
double rectangleRule();
double trapezoidalRule();

//////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
	const double ReferencePI = 3.141592653589793238462643; //ref value of pi for comparison
	int myID, nProcs;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myID);
	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
	if (myID == 0) {
		cout << "number of MPI processes = " << nProcs << endl;
	}

	// odd-even sort
	oddEvenSort();

	// numerical integration
	double p1 = rectangleRule();
	double p2 = trapezoidalRule();
	if (myID == 0) {
		cout << "rectangle rule  : pi = " << p1*4 << ", delta = " << abs(p1*4 - ReferencePI) << endl;
		cout << "trapezoidal rule: pi = " << p2*4 << ", delta = " << abs(p2*4 - ReferencePI) << endl;
	}

	MPI_Finalize();
	return 0;
}