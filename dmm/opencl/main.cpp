#include <CL/cl.hpp>

#include <iostream>
#include <string>
#include <vector>

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
	
	return 1;
}
