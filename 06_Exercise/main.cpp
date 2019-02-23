#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <omp.h>
#include <ppl.h>
#include "Stopwatch.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////
// global variables, prototypes
void quicksort(float a[], int left, int right);
void parallelQuicksort(float a[], int left, int right, int p);

////////////////////////////////////////////////////////////////////////////////////////
// comperator used in qsort
int compareTo(const void *a1, const void *a2) {
	double *s1 = (double *)a1;
	double *s2 = (double *)a2;
	
	if (*s1 < *s2) {
		return -1;
	} else if (*s1 > *s2) {
		return 1;
	} else {
		return 0;
	}
}

////////////////////////////////////////////////////////////////////////////////////////
// is array a sorted?
static bool isSorted(float a[], int n) {
	for (int i=0; i < n - 1; i++) {
		if (a[i] > a[i + 1]) return false;
	}
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////
static bool check(float *ref, float *sort, int n) {
	if (isSorted(sort, n)) {
		int i = 0;
		while(i < n && ref[i] == sort[i]) i++;
		if (i < n) {
			cout << "array containts wrong elements" << endl << endl;
			return false;
		} else {
			cout << "array is correctly sorted" << endl << endl;
			return true;
		}
	} else {
		cout << "array is not sorted" << endl << endl;
		return false;
	}
}

////////////////////////////////////////////////////////////////////////////////////////
static void print(float *data, int n) {
	for (int i=0; i < n - 1; i++) cout << data[i] << ',';
	cout << data[n - 1] << endl << endl;
}

////////////////////////////////////////////////////////////////////////////////////////
static void tests(int n, int p) {
	const size_t dataSize = n*sizeof(float);

	Stopwatch sw;
	float *data = new float[n];
	float *sortRef = new float[n];
	float *sort = new float[n];

	// init seed
	time_t now;
	time(&now);
	srand((int)now);
	
	// init arrays
	for (int i=0; i < n; i++) data[i] = (float)rand();
	memcpy(sortRef, data, dataSize);

	// omp settings
	omp_set_nested(true);
	cout << "Num Processors: " << omp_get_num_procs() << endl;
	cout << "Max Threads: " << omp_get_max_threads() << endl;
	cout << "Nested Threads: " << boolalpha << (omp_get_nested()) << endl << endl;

	// output small arrays
	if (n <= 20) print(data, n);

	// standard sequential qsort
	sw.Start();
	qsort(sortRef, n, sizeof(float), compareTo);
	sw.Stop();
	cout << "qsort (n = " << n << ") in " << sw.GetElapsedTimeMilliseconds() << " ms" << endl << endl;

	// stl sort
	memcpy(sort, data, dataSize);
	sw.Start();
	std::sort(sort, sort + n);
	sw.Stop();
	cout << "std::sort (n = " << n << ") in " << sw.GetElapsedTimeMilliseconds() << " ms" << endl;
	if (!check(sortRef, sort, n)) goto END;

	// standard parallel sort
	memcpy(sort, data, dataSize);
	sw.Start();
	Concurrency::parallel_sort(sort, sort + n);
	sw.Stop();
	cout << "parallel-sort (n = " << n << ") in " << sw.GetElapsedTimeMilliseconds() << " ms" << endl;
	if (!check(sortRef, sort, n)) goto END;

	// sequential quicksort
	memcpy(sort, data, dataSize);
	sw.Start();
	quicksort(sort, 0, n - 1);
	sw.Stop();
	cout << "quicksort (n = " << n << ") in " << sw.GetElapsedTimeMilliseconds() << " ms" << endl;
	if (!check(sortRef, sort, n)) goto END;

	// parallel quicksort
	memcpy(sort, data, dataSize);
	sw.Start();
	parallelQuicksort(sort, 0, n - 1, p);
	sw.Stop();
	cout << "parallel quicksort (n = " << n << ", p = " << p << ") in " << sw.GetElapsedTimeMilliseconds() << " ms" << endl;
	if (!check(sortRef, sort, n)) goto END;

	// output
	if (n <= 20) print(sort, n);

END:
	delete[] data;
	delete[] sortRef;
	delete[] sort;
}

////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char* argv[]) {
	if (argc < 2 || argc > 3) {
		cerr << "Usage: Quicksort n [p]" << endl
			 << "n: array length" << endl
			 << "p: number of processors (default = ld(n))" << endl;
		return -1;
	}

	int n = atoi(argv[1]);
	int p; 
	if (argc > 2) {
		p = atoi(argv[2]);
	} else {
		p = 1 + int(log(n)/log(2));
	}

	tests(n, p);

	// speed measurements
	for (int i = 15; i <= 27; i += 3) {
		tests(1 << i, 8);
	}

	return 0;
}