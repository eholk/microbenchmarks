#include <CL/cl.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <utility>

#include "dmm_kernel.h"
#include "../common.hpp"

using namespace std;
using namespace cl;

class OpenCLDmmBenchmark : public DmmBenchmark
{
		Context &ctx;
		CommandQueue &queue;
		Kernel &k;
public:
	struct Params {
		Params(Context &ctx, CommandQueue &&queue, Kernel &&k)
			: ctx(ctx), queue(queue), k(k)
		{}
		
		Context &ctx;
		CommandQueue &queue;
		Kernel &k;
	};
	
	OpenCLDmmBenchmark(int N, Params &params)
		: ctx(params.ctx), queue(params.queue), DmmBenchmark(N), k(params.k)
	{}

	virtual void run_iteration() {
		auto size = sizeof(float) * N * N;
		Buffer clA(ctx, CL_MEM_READ_ONLY, size),
			   clB(ctx, CL_MEM_READ_ONLY, size),
			   clC(ctx, CL_MEM_WRITE_ONLY, N * N * sizeof(A[0]));

		Event copy_a, copy_b, kernel, copy_c;
		
		queue.enqueueWriteBuffer(clA, false, 0, size, A, NULL, &copy_a);
		queue.enqueueWriteBuffer(clB, false, 0, size, B, NULL, &copy_b);

		k.setArg(0, N);
		k.setArg(1, clA);
		k.setArg(2, clB);
		k.setArg(3, clC);
		
		std::vector<Event> kernel_events = {copy_a, copy_b};
		queue.enqueueNDRangeKernel(k,
		                           NDRange(0, 0),
		                           NDRange(N, N),
		                           NullRange,
		                           &kernel_events,
		                           &kernel);
		std::vector<Event> copy_c_events = {kernel};
		queue.enqueueReadBuffer(clC, false, 0, size, C, &copy_c_events);
		queue.finish();
	}
};

int main(int argc, const char **argv) {
	cout << "name: opencl-dmm" << endl;

	// pick a platform
	std::vector<Platform> platforms;
	Platform::get(&platforms);
	cout << "# found " << platforms.size() << " platforms" << endl;
	auto platform = platforms[0];

	// display some information about the OpenCL platform
	cout << "opencl_platform: " << platform.getInfo<CL_PLATFORM_NAME>()
	     << endl;
	cout << "opencl_version: " << platform.getInfo<CL_PLATFORM_VERSION>()
	     << endl;		

	// pick a device
	std::vector<Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	cout << "# found " << devices.size() << " devices" << endl;

	auto device = devices[0];

	// display some information
	cout << "opencl_device: " << device.getInfo<CL_DEVICE_NAME>() << endl;
	cout << "device_type: "
	     << (device.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_CPU ?
	         "cpu" : "gpu")
	     << endl;

	// create and build the program.
	auto prog_devices = std::vector<Device>({device});
	Context ctx(prog_devices);
	std::vector<std::pair<const char*, ::size_t>> src = {
		make_pair((const char *)dmm_kernel_cl,
		          (::size_t)dmm_kernel_cl_len)
	};
	Program prog(ctx, src);
	prog.build(prog_devices);
	cout << "# built program" << endl;

	auto status = prog.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
	if(status != CL_BUILD_SUCCESS) {
		cerr << "# Program build failure!" << endl;
		cerr << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
	}

	run_benchmark<OpenCLDmmBenchmark>
		(OpenCLDmmBenchmark::Params(ctx,
		                            CommandQueue(ctx, device),
		                            Kernel(prog, "dmm")));
	
	return 0;
}
