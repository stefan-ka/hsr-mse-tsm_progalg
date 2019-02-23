#include "main.h"

////////////////////////////////////////////////////////////////////////
// Documentation
// http://www.openacc.org/sites/default/files/OpenACC.2.0a_1.pdf

////////////////////////////////////////////////////////////////////////
// RGB macros
#undef GetBValue
#undef GetGValue
#undef GetRValue
#undef RGB
#define GetBValue(rgb) (0xFF & rgb)
#define GetGValue(rgb) (0xFF & (rgb >> 8))
#define GetRValue(rgb) (0xFF & (rgb >> 16))
#define RGB(r, g, b)   ((((r << 8) | g) << 8) | b | 0xFF000000)

////////////////////////////////////////////////////////////////////////
static int dist(int x, int y) {
	int d = (int)sqrtf((float)(x*x) + (float)(y*y));
	return (d < 256) ? d : 255;
}

////////////////////////////////////////////////////////////////////////
void processACC(const fipImage& input, fipImage& output, const int *hFilter, const int *vFilter, int fSize) {
	const int bypp = 4;
	const int w = input.getWidth();
	const int h = input.getHeight();
	assert(w == output.getWidth() && h == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == bypp*8);
	const size_t stride = input.getScanWidth();
	const int fSizeD2 = fSize/2;
	const unsigned char* inputPtr = input.getScanLine(0);
	
	unsigned char* outputPtr = output.getScanLine(0);

	////////////////////////////////////////////////////////////////////////
	// TODO implement edge detection with OpenACC
	// access images by inputPtr and outputPtr
	// use const COLORREF c = *reinterpret_cast<const COLORREF*>(inputPtr) to read the pixel color c, and use GetBValue(c) to access the blue channel of color c
	// use *reinterpret_cast<COLORREF*>(outputPtr) = RGB(r, g, b) to write a pixel with RGB channels to output image
	// parallelize loops with #pragma acc ...

}
