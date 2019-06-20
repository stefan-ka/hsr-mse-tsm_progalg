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

static void pQsort(float a[], float b[], int left, int right, int p) {
	if (left >= right) 
		return;

	assert(a);
	assert(p > 0);

	if (p == 1) {
		quicksort(a, left, right);
	} else {
		assert(p > 1);
		const int pp = p + 1;
		const int n = right + 1;

		int *s = new int[pp]; s[p] = n; // start indices per thread
		int *l = new int[p];			// index positions of the elements larger than pivot
		int *q = new int[pp];			// left + prefix sums of the number of elements <= pivot
		int *r = new int[pp];			// prefix sums of the number of elements > pivot

		// compute pivot position
		#ifdef _RANDOMPIVOT_
		int pivotPos = random(left, right);
		#else
		int pivotPos = median(a, left, left + (right - left) / 2, right);
		#endif
		const float pivot = a[pivotPos];

		// reposition pivot
		assert(a[pivotPos] == pivot);
		a[pivotPos] = a[left]; a[left] = pivot;
		pivotPos = left;

		#pragma omp parallel default(none) shared(s, l) num_threads(p)
		{
			// local rearrangement

			// init s and l in parallel: O(1)
			const int index = omp_get_thread_num();
			s[index] = l[index] = n;

			// compute s and locally rearrange in parallel: O(n/p)
			#pragma omp for
			for (int i = left; i <= right; i++) {
				if (s[index] == n) {
					s[index] = l[index] = i;
				}
				if (a[i] <= pivot) {
					float tmp = a[l[index]]; a[l[index]++] = a[i]; a[i] = tmp;
				}
			}

			// compute prefix-sums q and r: O(p), simple but not optimal
			#pragma omp single
			{
				q[0] = left;
				r[0] = 0;
				for (int i = 1; i <= p; i++) {
					q[i] = q[i - 1] + (l[i - 1] - s[i - 1]);
					r[i] = r[i - 1] + (s[i] - l[i - 1]);
				}

				assert(a[pivotPos] == pivot);
			}

			// global rearrangement

			// copy to b and rearrange in parallel: O(n/p)
			#pragma omp for
			for (int i = 0; i < p; i++) {
				int d = l[i] - s[i];
				for (int k = 0; k < d; k++) {
					b[q[i] + k] = a[s[i] + k];
				}
				d = s[i + 1] - l[i];
				for (int k = 0; k < d; k++) {
					b[q[p] + r[i] + k] = a[l[i] + k];
				}
			}
		}

		// copy from b to a: O(n), simpler and faster than parallel copying
		memcpy(a + left, b + left, (right - left + 1) * sizeof(float));

		// splitting position
		int m = q[p] - 1;

		// place pivot
		assert(a[pivotPos] == pivot);
		a[pivotPos] = a[m]; a[m] = pivot;

		// partition processes
		int p1 = __max(1, p*(m - left) / (right - left));

		delete[] s;
		delete[] l;
		delete[] q;
		delete[] r;

		#pragma omp parallel sections default(none) num_threads(2)
		{
			#pragma omp section
			pQsort(a, b, left, m - 1, p1);
			#pragma omp section
			pQsort(a, b, m + 1, right, p - p1);
		}
	}
}

void pQSortSimple(float a[], int left, int right) {
	if (left < right) {
		// compute pivot
		#ifdef _RANDOMPIVOT_
		int pivotPos = random(left, right);
		#else
		int pivotPos = median(a, left, left + (right - left) / 2, right);
		#endif
		const float pivot = a[pivotPos];

		// reposition pivot
		assert(a[pivotPos] == pivot);
		a[pivotPos] = a[left]; a[left] = pivot;
		pivotPos = left;

		// partition
		int s = pivotPos;
		for (int i = left + 1; i <= right; i++) {
			if (a[i] <= pivot) {
				s++;
				float tmp = a[s]; a[s] = a[i]; a[i] = tmp;
			}
		}
		float tmp = a[pivotPos]; a[pivotPos] = a[s]; a[s] = tmp;

		#pragma omp parallel sections default(none) num_threads(2)
		{
			#pragma omp section
			pQSortSimple(a, left, s - 1);
			#pragma omp section
			pQSortSimple(a, s + 1, right);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////
// parallel quicksort
// sorts a[left]..a[right] using p threads 
void parallelQuicksort(float a[], int left, int right, int p) {
	assert(p > 0);
	assert(left >= 0 && left <= right);
	const int n = right + 1;
	float *b = new float[n]; // temporary storage

	memset(b, 0, n * sizeof(float));

	pQsort(a, b, left, right, p);
	delete[] b;
}
