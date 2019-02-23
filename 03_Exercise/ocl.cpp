#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include "main.h"
#include "ocl.h"

////////////////////////////////////////////////////////////////////////
OCLData initOCL(const char* kernelFileName, const char* kernelName) {
	OCLData ocl;
	vector<cl::Platform> platforms;
	vector<cl::Device> devices;

	cout << "**************************************************************************" << endl;
	try {
		// discover all available platforms
		cl::Platform::get(&platforms);
		if (platforms.empty()) {
			cerr << "Error: no OpenCL platform available" << endl;
			return ocl;
		}

		string name, profile, version;
		
		for(cl::Platform& p: platforms) {
			p.getInfo(CL_PLATFORM_NAME, &name);
			p.getInfo(CL_PLATFORM_PROFILE, &profile);
			p.getInfo(CL_PLATFORM_VERSION, &version);

			vector<cl::Device> devs;

			try {
				// get GPUs of this platform
				p.getDevices(CL_DEVICE_TYPE_GPU, &devs);

			} catch(cl::Error&) {
				// there is no GPU on this platform
			}

			if (!devs.empty()) {
				// show platform
				cout << "Platform " << name << " (" << version << ", " << profile << ")" << endl;

				// discover all available GPUs on this platform
				for(cl::Device& dev: devs) {
					// get and show device information
					dev.getInfo(CL_DEVICE_NAME, &name);
					cl_uint maxComputeUnits;		dev.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &maxComputeUnits);
					cl_uint maxDims;				dev.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &maxDims);
					vector<size_t> workitemSize;	dev.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &workitemSize);
					size_t workgroupSize;			dev.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &workgroupSize); // product of all dimensions
					cl_ulong localMemSize;			dev.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &localMemSize);
					cout << name << endl;
					cout << "Max. Compute Units: " << maxComputeUnits << ", Max. Dimensions: " << maxDims << " [" << workitemSize[0] << "," << workitemSize[1] << "," << workitemSize[2] << "]" << endl;
					cout << "Max. Work Group Size: " << workgroupSize << ", Max. Local Mem Size: " << localMemSize << endl << endl;
				}

				if (devices.empty()) {
					// create s_context and command s_queue for the first GPU
					cl::Device& dev = devs.front();
					cl_context_properties cps[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(p)(), 0 };

					ocl.m_context = cl::Context(CL_DEVICE_TYPE_GPU, cps);
					ocl.m_queue = cl::CommandQueue(ocl.m_context, dev);

					// add first device
					devices.push_back(dev);
				}
			}
		}

		// read source file
		ifstream file(kernelFileName);
		if (!file.good()) {
			perror("Error");
			return ocl;
		}
		string prog(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));	// read in file and store it in string prog
		file.close();
		cl::Program::Sources progSource(1, make_pair(prog.c_str(), prog.length() + 1));	// create source code object

		// compile and execute s_kernel code
		cl::Program program(ocl.m_context, progSource);									// create program object
		program.build(devices);															// build the program for given devices
		ocl.m_kernel = cl::Kernel(program, kernelName);									// create the s_kernel: must be the name of the s_kernel in the cl file

	} catch(cl::Error& err) {
		cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")" << endl;
	}
	cout << "**************************************************************************" << endl;
	return ocl;
}

////////////////////////////////////////////////////////////////////////
void processOCL(OCLData& ocl, const fipImage& input, fipImage& output, const int *hFilter, const int *vFilter, int fSize) {
	const int bypp = 4;
	const size_t w = input.getWidth();
	const size_t h = input.getHeight();
	assert(w == output.getWidth() && h == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == bypp*8);
	const size_t stride = input.getScanWidth();
	const int fSize2 = fSize*fSize;
	
	cl::size_t<3> origin;
	cl::size_t<3> region; 
	region[0] = w; 
	region[1] = h; 
	region[2] = 1;

	try {
		// the image format describes the properties of each pixel
		cl::ImageFormat format;
		format.image_channel_order = CL_BGRA;
		format.image_channel_data_type = CL_UNSIGNED_INT8;

		// create sampler object
		cl::Sampler sampler(ocl.m_context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST); // on CPU must be not CL_ADDRESS_NONE

		// create space for the images
		cl::Image2D source(ocl.m_context, CL_MEM_READ_ONLY, format, region[0], region[1], 0);
		cl::Image2D dest (ocl.m_context, CL_MEM_WRITE_ONLY, format, region[0], region[1], 0);

		// create space for the filters
		cl::Buffer hor(ocl.m_context, CL_MEM_READ_ONLY, fSize2*sizeof(int));
		cl::Buffer ver(ocl.m_context, CL_MEM_READ_ONLY, fSize2*sizeof(int));

		// write buffers to device
		ocl.m_queue.enqueueWriteImage(source, CL_TRUE, origin, region, stride, 0, input.getScanLine(0));
		ocl.m_queue.enqueueWriteBuffer(hor, CL_TRUE, 0, fSize2*sizeof(int), hFilter);
		ocl.m_queue.enqueueWriteBuffer(ver, CL_TRUE, 0, fSize2*sizeof(int), vFilter);

		// set the s_kernel arguments
		ocl.m_kernel.setArg(0, source);
		ocl.m_kernel.setArg(1, dest);
		ocl.m_kernel.setArg(2, hor);
		ocl.m_kernel.setArg(3, ver);
		ocl.m_kernel.setArg(4, fSize);
		ocl.m_kernel.setArg(5, sampler);

		// run the kernels
		ocl.m_queue.enqueueNDRangeKernel(ocl.m_kernel, cl::NullRange, cl::NDRange(region[0], region[1]), cl::NullRange);

		// read the output buffer back to the host
		ocl.m_queue.enqueueReadImage(dest, CL_TRUE, origin, region, stride, 0, output.getScanLine(0));

	} catch(cl::Error& err) {
		cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")" << endl;
	}
}
