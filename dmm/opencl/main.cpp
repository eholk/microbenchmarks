#include <CL/cl.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <utility>

#include "dmm_kernel.h"

using namespace std;
using namespace cl;

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
	
	return 1;
}
