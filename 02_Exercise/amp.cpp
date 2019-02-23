#include <amp.h>
#include <amp_math.h>
#include "main.h"

////////////////////////////////////////////////////////////////////////
// Documentation
// http://msdn.microsoft.com/de-de/library/hh265137.aspx
// https://www.microsoftpressstore.com/articles/article.aspx?p=2201645

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
static int dist(int x, int y) restrict(amp) {
#ifdef FAST_MATH
	int d = (int)Concurrency::fast_math::sqrtf((float)(x*x) + (float)(y*y));
#else
	int d = (int)Concurrency::precise_math::sqrtf((float)(x*x) + (float)(y*y));
#endif
	return (d < 256) ? d : 255;
}

////////////////////////////////////////////////////////////////////////
void processAMP(const fipImage& input, fipImage& output, const int *hFilter, const int *vFilter, int fSize, Stopwatch& sw) {
	const int bypp = 4;
	const int w = input.getWidth();
	const int h = input.getHeight();
	assert(w == output.getWidth() && h == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == bypp*8);
	const unsigned int stride = input.getScanWidth();
	const int fSizeD2 = fSize/2;

	// list accelerators
	std::vector<Concurrency::accelerator> accls = Concurrency::accelerator::get_all();

	std::wcout << "Found " << accls.size() << " C++ AMP accelerator(s):" << std::endl;
	std::for_each(accls.cbegin(), accls.cend(), [](const Concurrency::accelerator& a)
	{
		std::wcout << "  " << a.device_path << std::endl << "    " << a.description << std::endl << std::endl;
	});
	std::wcout << "AMP default accelerator: " << Concurrency::accelerator(Concurrency::accelerator::default_accelerator).description << std::endl;

	sw.Start();
	
	// create array-views: they manage data transport between host and GPU
	Concurrency::array_view<const COLORREF, 2> avI(stride/bypp, h, reinterpret_cast<COLORREF*>(input.getScanLine(0)));
	Concurrency::array_view<COLORREF, 2> avR(stride/bypp, h, reinterpret_cast<COLORREF*>(output.getScanLine(0)));
	Concurrency::array_view<const int, 2> avH(fSize, fSize, hFilter);
	Concurrency::array_view<const int, 2> avV(fSize, fSize, vFilter);

	// parallel processing on GPU
	Concurrency::parallel_for_each(avI.extent, [=](Concurrency::index<2> idx) restrict(amp) {
		// TODO implement convolution
		// use COLORREF c = avI[idx] to read a pixel, and use GetBValue(c) to access the blue channel of color c
		// use avR[idx] = RGB(r, g, b) to write a pixel with RGB channels to output image

	});

	// wait until the GPU has finished and the result has been copied back to output
	avR.synchronize();

	sw.Stop();
}
