#include "Stopwatch.h"
#include "main.h"
#include "ocl.h"

////////////////////////////////////////////////////////////////////////
// prototypes
OCLData initOCL(const char* kernelFileName, const char* kernelName, const bool useCPU);
void processOCL(OCLData& ocl, const fipImage& input, fipImage& output, const int *hFilter, const int *vFilter, int fSize);

////////////////////////////////////////////////////////////////////////
static BYTE dist(int x, int y) {
	int d = (int)sqrtf((float)(x*x) + (float)(y*y));
	return (d < 256) ? d : 255;
}

////////////////////////////////////////////////////////////////////////
static void processParallel(const fipImage& input, fipImage& output, const int *hFilter, const int *vFilter, int fSize) {
	const int bypp = 4;
	assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == bypp*8);

	const size_t stride = input.getScanWidth();
	const int fSizeD2 = fSize/2;

	#pragma omp parallel for
	for(int v = fSizeD2; v < (int)output.getHeight() - fSizeD2; v++) {
		BYTE *iCenter = input.getScanLine(v) + bypp*fSizeD2;
		BYTE *oPos = output.getScanLine(v) + bypp*fSizeD2;

		for(size_t u = fSizeD2; u < output.getWidth() - fSizeD2; u++) {
			int hC[3] = { 0, 0, 0 };
			int vC[3] = { 0, 0, 0 };
			int fi = 0;
			BYTE *iPos = iCenter - fSizeD2*stride - bypp*fSizeD2;

			for(int j = 0; j < fSize; j++) {
				for(int i = 0; i < fSize; i++) {
					const RGBQUAD *iC = reinterpret_cast<RGBQUAD*>(iPos);
					int f = hFilter[fi];

					hC[0] += f*iC->rgbBlue;
					hC[1] += f*iC->rgbGreen;
					hC[2] += f*iC->rgbRed;
					f = vFilter[fi];
					vC[0] += f*iC->rgbBlue;
					vC[1] += f*iC->rgbGreen;
					vC[2] += f*iC->rgbRed;
					iPos += bypp;
					fi++;
				}
				iPos += stride - bypp*fSize;
			}
			RGBQUAD *oC = reinterpret_cast<RGBQUAD*>(oPos);
			oC->rgbBlue = dist(hC[0], vC[0]);
			oC->rgbGreen = dist(hC[1], vC[1]);
			oC->rgbRed = dist(hC[2], vC[2]);
			oC->rgbReserved = 255;
			iCenter += bypp;
			oPos += bypp;
		}
	}
}

////////////////////////////////////////////////////////////////////////
static bool equals(const fipImage& im1, const fipImage& im2, int fSize) {
	assert(im1.getWidth() == im2.getWidth() && im1.getHeight() == im2.getHeight() && im1.getImageSize() == im2.getImageSize());
	assert(im1.getBitsPerPixel() == 32);
	const int fSizeD2 = fSize/2;

	for(unsigned int i = fSizeD2; i < im1.getHeight() - fSizeD2; i++) {
		COLORREF *row1 = reinterpret_cast<COLORREF*>(im1.getScanLine(i));
		COLORREF *row2 = reinterpret_cast<COLORREF*>(im2.getScanLine(i));
		for(unsigned int j = fSizeD2; j < im1.getWidth() - fSizeD2; j++) {
			if (row1[j] != row2[j]) return false;
		}
	}
	return true;
}

////////////////////////////////////////////////////////////////////////
int main(int argc, const char* argv[]) {
	if (argc < 4) {
		cerr << "Usage: " << argv[0] << " filter-size input-file-name output-file-name" << endl;
		return -1;
	}
	int fSize = atoi(argv[1]);
	if (fSize < 3 || fSize > 11 || (fSize & 1) == 0) {
		cerr << "Wrong filter size. Filter size must be odd and between 3 and 9" << endl;
		return -2;
	}

	fipImage image;

	// load image
	if (!image.load(argv[2])) {
		cerr << "Image not found: " << argv[2] << endl;
		return -3;
	}

	const int hFilter3[] = {
		1, 1, 1,
		0, 0, 0,
	   -1,-1,-1,
	};
	const int vFilter3[] = {
		1, 0,-1,
		1, 0,-1,
		1, 0,-1,
	};
	const int hFilter5[] = {
		0, 0, 0, 0, 0,
		1, 1, 1, 1, 1,
		0, 0, 0, 0, 0,
	   -1,-1,-1,-1,-1,
	    0, 0, 0, 0, 0,
	};
	const int vFilter5[] = {
		0, 1, 0,-1, 0,
		0, 1, 0,-1, 0,
		0, 1, 0,-1, 0,
		0, 1, 0,-1, 0,
		0, 1, 0,-1, 0,
	};
	const int hFilter7[] = {
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		1, 1, 1, 1, 1, 1, 1,
		0, 0, 0, 0, 0, 0, 0,
	   -1,-1,-1,-1,-1,-1,-1,
	    0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
	};
	const int vFilter7[] = {
		0, 0, 1, 0,-1, 0, 0,
		0, 0, 1, 0,-1, 0, 0,
		0, 0, 1, 0,-1, 0, 0,
		0, 0, 1, 0,-1, 0, 0,
		0, 0, 1, 0,-1, 0, 0,
		0, 0, 1, 0,-1, 0, 0,
		0, 0, 1, 0,-1, 0, 0,
	};
	const int hFilter9[] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 1, 1, 1, 1, 1, 1, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
	   -1,-1,-1,-1,-1,-1,-1,-1,-1,
	    0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
	};
	const int vFilter9[] = {
		0, 0, 0, 1, 0,-1, 0, 0, 0,
		0, 0, 0, 1, 0,-1, 0, 0, 0,
		0, 0, 0, 1, 0,-1, 0, 0, 0,
		0, 0, 0, 1, 0,-1, 0, 0, 0,
		0, 0, 0, 1, 0,-1, 0, 0, 0,
		0, 0, 0, 1, 0,-1, 0, 0, 0,
		0, 0, 0, 1, 0,-1, 0, 0, 0,
		0, 0, 0, 1, 0,-1, 0, 0, 0,
		0, 0, 0, 1, 0,-1, 0, 0, 0,
	};
	const int hFilter11[] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	   -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
	    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	};
	const int vFilter11[] = {
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
	};

	Stopwatch sw;
	double parTime;
	const int *hFilter;
	const int *vFilter;

	switch(fSize) {
	case 3:
		hFilter = hFilter3;
		vFilter = vFilter3;
		break;
	case 5:
		hFilter = hFilter5;
		vFilter = vFilter5;
		break;
	case 7:
		hFilter = hFilter7;
		vFilter = vFilter7;
		break;
	case 9:
		hFilter = hFilter9;
		vFilter = vFilter9;
		break;
	case 11:
		hFilter = hFilter11;
		vFilter = vFilter11;
		break;
	}

	// create output images
	fipImage out1(image), out2(image), out3(image), out4(image);

	cout << "Edge detection with filter size " << fSize << endl << endl;

	// process image on CPU in parallel and produce out1
	cout << "Start OpenMP" << endl;
	sw.Start();
	processParallel(image, out1, hFilter, vFilter, fSize);
	sw.Stop();
	parTime = sw.GetElapsedTimeMilliseconds();
	cout << parTime << " ms" << endl << endl;
	
	// process image on GPU with OpenCL and produce out2
	OCLData ocl = initOCL("edges.cl", "edges", false);
	cout << endl << "Start OpenCL on GPU" << endl;
	sw.Start();
	processOCL(ocl, image, out2, hFilter, vFilter, fSize);
	sw.Stop();
	cout << sw.GetElapsedTimeMilliseconds() << " ms, speedup = " << parTime/sw.GetElapsedTimeMilliseconds() << endl;

	// compare out1 with out2
	cout << boolalpha << "OpenMP and OpenCL on GPU produce the same results: " << equals(out1, out2, fSize) << endl << endl;

	OCLData oclCPU = initOCL("edges.cl", "edges", true);
	cout << endl << "Start OpenCL on CPU" << endl;
	sw.Start();
	processOCL(oclCPU, image, out3, hFilter, vFilter, fSize);
	sw.Stop();
	cout << sw.GetElapsedTimeMilliseconds() << " ms, speedup = " << parTime / sw.GetElapsedTimeMilliseconds() << endl;

	// compare out1 with out3
	cout << boolalpha << "OpenMP and OpenCL on CPU produce the same results: " << equals(out1, out3, fSize) << endl << endl;

	// save output image
	if (!out3.save(argv[3])) {
		cerr << "Image not saved: " << argv[3] << endl;
		return -1;
	}

	return 0;
}
