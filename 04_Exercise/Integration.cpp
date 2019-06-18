#include <time.h>
#include <mpi.h> 
#include <iostream>

using namespace std;

// the function to be integrated: f(x)= 1 / (1 + x*x)
static double integF(double x) {
	return 1.0 / (1.0 + x * x);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// num integration in the domain [0,1] of f(x)= 1 / (1 + x*x)
// midpoint or rectangle rule
double rectangleRule() {
	int nproc, myid, intervals;
	
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	if (myid == 0) {
		cout << "Please enter the number of intervals: ";
		cin >> intervals;
	}

	MPI_Bcast(&intervals, 1, MPI_INT, 0, MPI_COMM_WORLD);

	const double interval_size = 1.0 / intervals;
	double localSum = 0.0;
	for (int interval = myid + 1; interval <= intervals; interval += nproc) {
		localSum += interval_size * integF(interval_size * (interval - 0.5));
	}
	
	double sum = 0;
	MPI_Reduce(&localSum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	return sum;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// num integration in the domain [0,1] of f(x)= 1 / (1 + x*x)
// trapezoidal rule
double trapezoidalRule() {
	int nproc, myid, intervals;

	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	if (myid == 0) {
		cout << "Please enter the number of intervals: ";
		cin >> intervals;
	}

	MPI_Bcast(&intervals, 1, MPI_INT, 0, MPI_COMM_WORLD);

	const int block = (intervals + nproc - 1) / nproc;
	const double interval_size = 1.0 / intervals;
	const double x0 = myid * block * interval_size;
	double localSum = (x0 < 1) ? ((myid == 0) ? 0.5 / (1.0 + x0 * x0) : 1.0 / (1.0 + x0 * x0)) : 0;
	
	for (int i = 1; i < block; i++) {
		double x = x0 + i * interval_size;
		if (x < 1)
			localSum += integF(x);
	}

	if (myid == nproc - 1) {
		localSum += 0.25;
	}

	double sum = 0;
	localSum = interval_size * localSum;
	MPI_Reduce(&localSum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	return sum;
}
