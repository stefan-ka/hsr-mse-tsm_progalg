#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <cassert>
#include "ocl.h"

using namespace std;

////////////////////////////////////////////////////////////////////////
const int devType = CL_DEVICE_TYPE_GPU;

////////////////////////////////////////////////////////////////////////
OCLData initOCL(const char* kernelFileName, const char* kernelName) {
	OCLData ocl;
	vector<cl::Platform> platforms;
	vector<cl::Device> devices;

	//cout << "**************************************************************************" << endl;
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
				p.getDevices(devType, &devs);

			} catch(cl::Error&) {
				// there is no GPU on this platform
			}

			if (!devs.empty()) {
				// show platform
				//cout << "Platform " << name << " (" << version << ", " << profile << ")" << endl;

				// discover all available GPUs on this platform
				for(cl::Device& dev: devs) {
					// get and show device information
					cl_uint maxComputeUnits;		dev.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &maxComputeUnits);
					cl_uint maxDims;				dev.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &maxDims);
					vector<size_t> workitemSize;	dev.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &workitemSize);
					size_t workgroupSize;			dev.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &workgroupSize); // product of all dimension
					cl_ulong localMemSize;			dev.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &localMemSize);
					//cout << "Max. Compute Units: " << maxComputeUnits << ", Max. Dimensions: " << maxDims << " [" << workitemSize[0] << "," << workitemSize[1] << "," << workitemSize[2] << "]" << endl;
					//cout << "Max. Work Group Size: " << workgroupSize << ", Max. Local Mem Size: " << localMemSize << endl << endl;
				}

				if (devices.empty()) {
					// create s_context and command s_queue for the first device
					cl_context_properties cps[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(p)(), 0 };
					ocl.m_context = cl::Context(devType, cps);
					ocl.m_queue = cl::CommandQueue(ocl.m_context, devs[0]);

					// add first device
					devices.push_back(devs[0]);
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
	#ifdef _DEBUG
		if (devType == CL_DEVICE_TYPE_CPU) {
			string s;
			s.append("-g -s \"").append(kernelFileName).append("\"");
			program.build(devices, s.c_str());											// start debugger
		} else
	#endif
		{
			program.build(devices);														// build the program for given devices
		}
		ocl.m_kernel = cl::Kernel(program, kernelName);									// create the s_kernel: must be the name of the s_kernel in the cl file

		// get device infos
		devices[0].getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &ocl.m_computeUnits);

	} catch(cl::Error& err) {
		cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")" << endl;
	}
	//cout << "**************************************************************************" << endl;
	return ocl;
}

////////////////////////////////////////////////////////////////////////
void matMultGPU(OCLData& ocl, const int* a, const int* b, int* const c, const int n) {
	// TODO
	// do data exchange between CPU and GPU and start the GPU kernel 
}