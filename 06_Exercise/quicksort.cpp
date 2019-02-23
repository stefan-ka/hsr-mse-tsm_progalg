#include <cassert>
#include <omp.h>
#include <cstdlib>
#include <algorithm>

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////
// compiler directives
#undef _RANDOMPIVOT_

////////////////////////////////////////////////////////////////////////////////////////
// returns a random integer in range [a,b]
static int random(int a, int b) {
	assert(b >= a);

	const int n = b - a;
	int ret;

	if (n > RAND_MAX) {
		ret = a + ((n/RAND_MAX)*rand() + rand())%(n + 1);
	} else {
		ret = a + rand()%(n + 1);
	}
	assert(a <= ret && ret <= b);
	return ret;
}

////////////////////////////////////////////////////////////////////////////////////////
// determine median of a[p1], a[p2], and a[p3]
static int median(float a[], int p1, int p2, int p3) {
	float ap1 = a[p1];
	float ap2 = a[p2];
	float ap3 = a[p3];

	if (ap1 <= ap2) {
		return (ap2 <= ap3) ? p2 : ((ap1 <= ap3) ? p3 : p1);
	} else {
		return (ap1 <= ap3) ? p1 : ((ap2 <= ap3) ? p3 : p2);
	}
}

////////////////////////////////////////////////////////////////////////////////////////
// serial quicksort
// sorts a[left]..a[right]
void quicksort(float a[], int left, int right) {
	// compute pivot
#ifdef _RANDOMPIVOT_
	int pivotPos = random(left, right);
#else
	int pivotPos = median(a, left, left + (right - left)/2, right);
#endif
	const float pivot = a[pivotPos];

	int i = left, j = right;

	do {
		while (a[i] < pivot) i++;
		while (pivot < a[j]) j--;
		if (i <= j) {
			swap(a[i], a[j]);
			i++;
			j--;
		}
	} while (i < j);
	if (left < j) quicksort(a, left, j);
	if (i < right) quicksort(a, i, right);
}

////////////////////////////////////////////////////////////////////////////////////////
// parallel quicksort
// sorts a[left]..a[right] using p threads 
void parallelQuicksort(float a[], int left, int right, int p) {
	assert(p > 0);
	assert(left >= 0 && left <= right);

	// TODO
}

