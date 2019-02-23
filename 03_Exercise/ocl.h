#pragma once

#include <iostream>
#include <cassert>
#include "FreeImagePlus.h"
#include <cvmarkersobj.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

using namespace std;
using namespace Concurrency::diagnostic;

#ifndef WIN32
typedef unsigned long COLORREF;
#endif

string getPath(const char path[], int k = 1);

struct OCLData {
	cl::Context m_context;
	cl::CommandQueue m_queue;
	cl::Kernel m_kernel;
};

struct OCLDataCPU : public OCLData {
	cl::Context m_contextC;
	cl::CommandQueue m_queueC;
	cl::Kernel m_kernelC;
};