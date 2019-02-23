#pragma once

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

// http://www.khronos.org/registry/cl/specs/opencl-cplusplus-1.2.pdf	// C++ manual
// http://www.khronos.org/registry/cl/specs/opencl-1.2.pdf				// C manual

// error numbers are defined in cl.h 
//#include <CL/cl.h>

struct OCLData {
	cl::Context m_context;
	cl::CommandQueue m_queue;
	cl::Kernel m_kernel;
};