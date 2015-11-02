#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <mach/mach_time.h>
#else
#include <CL/opencl.h>
#endif

#include <stdio.h>
#include <inttypes.h>
#include <time.h>

void handle_error(cl_int e, const char * file, int line);
void real_check_status(cl_int e, const char *file, int line);
void print_vector(float *x, int len);
uint64_t time_ns();

#define check_status(e) real_check_status(e, __FILE__, __LINE__);

cl_platform_id g_platform;
cl_device_id g_device;
cl_context g_context;
cl_command_queue g_queue;

cl_device_id find_device(cl_device_type type) {
    cl_int status;

    cl_platform_id *platforms = NULL;
    cl_uint nPlatforms = 0;

    cl_platform_id platform;
    cl_device_id device;

    // Call once with NULL to determine how much space we need to
    // allocate.
    status = clGetPlatformIDs(0, NULL, &nPlatforms);
    check_status(status);

    fprintf(stderr, "# Found %d platforms.\n", nPlatforms);
    

    // Allocate space for the platform IDs.
    platforms = (cl_platform_id *)calloc(nPlatforms, sizeof(cl_platform_id));

    // Get the platform IDs.
    status = clGetPlatformIDs(nPlatforms, platforms, &nPlatforms);
    check_status(status);

    // Try each platform until we find a device.
    for(int i = 0; i < nPlatforms; ++i) {
        fprintf(stderr, "# Trying platform %d.\n", i);
        // Pick the first platform.
        cl_platform_id platform = platforms[i];

        // Find out how many devices there are.
        cl_uint n_dev = 0;
        status = clGetDeviceIDs(platform, type, 0, NULL, &n_dev);

        if(CL_DEVICE_NOT_FOUND == status) {
            fprintf(stderr, "# No devices found on platform %d.\n", i);
            continue;
        }

        check_status(status);
        
        fprintf(stderr, "# Found %d devices on platform %d.\n", n_dev, i);
        
        // Allocate space for the device IDs
        cl_device_id *devices = NULL;
        devices = (cl_device_id *)calloc(n_dev, sizeof(cl_device_id));

        // Get the device IDs
        status = clGetDeviceIDs(platform, type, n_dev,
                                devices, &n_dev);
        check_status(status);

        // Arbitrarily pick the first device.
        device = devices[0];
        free(devices);
        break;
    }

    free(platforms);

    return device;
}

int main() {
    cl_int status;

    printf("name: clmemcopy\n");
    
    // Find the platforms
    g_device = find_device(CL_DEVICE_TYPE_GPU
                           | CL_DEVICE_TYPE_ACCELERATOR);


    char n[256];
    clGetDeviceInfo(g_device, CL_DEVICE_NAME, sizeof(n), n, NULL);

    printf("device: %s\n", n);

    
// Create a context for the devices.
    g_context = clCreateContext(0, 1, &g_device,
                                NULL, // This could be a pointer to a
                                      // notify function.

                                NULL, // And this could be a pointer
                                      // to some application specific
                                      // data that is passed to the
                                      // notify function.
                                &status);
    check_status(status);

    // Now we'll set up some vectors to get ready for the kernel.
    const int SIZE = 256 << 20; // 256MB

    // And create memory buffers.
    void *cpu = malloc(SIZE);

    cl_mem gpu = clCreateBuffer(g_context,
                                CL_MEM_READ_ONLY,
                                SIZE,
                                NULL, // host pointer...
                                &status);
    check_status(status);

	// Create the command queue
	g_queue =
	  clCreateCommandQueue(g_context,
						   g_device,
						   0, // flags, such as
						   // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLED
						   &status);
	check_status(status);

    const int REPS = 10;

    // Do an initial copy to warm up.
    status = clEnqueueWriteBuffer(g_queue,
                                  gpu,
                                  CL_TRUE, // blocking
                                  0, // offset
                                  1,
                                  cpu,
                                  0, // no events in wait list
                                  NULL, // no wait list
                                  NULL); // We'll ignore the
                                         // returned event.
    check_status(status);

    printf("results:\n");
    
    for(int i = 1; i <= SIZE; i <<= 1) {
	    printf("- byte_size: %d\n", i);
	    printf("  raw_data:\n");
	    printf("    total_time:\n");
        for(int j = 0; j < REPS; j++) {
	        uint64_t start = time_ns();
            // Copy the data
            status = clEnqueueWriteBuffer(g_queue,
                                          gpu,
                                          CL_TRUE, // blocking
                                          0, // offset
                                          i,
                                          cpu,
                                          0, // no events in wait list
                                          NULL, // no wait list
                                          NULL); // We'll ignore the
                                                 // returned event.
            check_status(status);
            uint64_t stop = time_ns();
            printf("    - %f\n", ((double)(stop - start)) / 1e9);
        }
    }

	// Clean up
	clReleaseCommandQueue(g_queue);
    clReleaseContext(g_context);
	clReleaseMemObject(gpu);
    free(cpu);
    return 0;
}

void print_vector(float *x, int len) {
  printf("[");
  for(int i = 0; i < len; ++i) {
	printf(" %f", x[i]);
  }
  printf(" ]\n");
}

void real_check_status(cl_int e, const char *file, int line) {
    if(CL_SUCCESS != e) {
        handle_error(e, file, line);
    }
}

void handle_error(cl_int e, const char *file, int line) {
#define HANDLE(x)                                                       \
    if(e == x) {                                                        \
        fprintf(stderr, "OpenCL error in %s, line %d: " #x " (%d)\n",   \
                file, line, e);                                         \
		abort();														\
    }
 
    HANDLE(CL_BUILD_PROGRAM_FAILURE);
    HANDLE(CL_COMPILER_NOT_AVAILABLE);
    HANDLE(CL_DEVICE_NOT_FOUND);
	HANDLE(CL_INVALID_ARG_INDEX);
	HANDLE(CL_INVALID_ARG_SIZE);
	HANDLE(CL_INVALID_ARG_VALUE);
    HANDLE(CL_INVALID_BINARY);
    HANDLE(CL_INVALID_BUILD_OPTIONS);
    HANDLE(CL_INVALID_COMMAND_QUEUE);
    HANDLE(CL_INVALID_CONTEXT);
    HANDLE(CL_INVALID_DEVICE);
    HANDLE(CL_INVALID_DEVICE_TYPE);
    HANDLE(CL_INVALID_EVENT_WAIT_LIST);
    HANDLE(CL_INVALID_GLOBAL_OFFSET);
    //HANDLE(CL_INVALID_GLOBAL_WORK_SIZE);
    HANDLE(CL_INVALID_IMAGE_SIZE);
    HANDLE(CL_INVALID_KERNEL);
    HANDLE(CL_INVALID_KERNEL_ARGS);
	HANDLE(CL_INVALID_MEM_OBJECT);
    HANDLE(CL_INVALID_OPERATION);
    HANDLE(CL_INVALID_PLATFORM);
    HANDLE(CL_INVALID_PROGRAM);
    HANDLE(CL_INVALID_PROGRAM_EXECUTABLE);
    HANDLE(CL_INVALID_QUEUE_PROPERTIES);
	HANDLE(CL_INVALID_SAMPLER);
    HANDLE(CL_INVALID_VALUE);
    HANDLE(CL_INVALID_WORK_DIMENSION);
    HANDLE(CL_INVALID_WORK_GROUP_SIZE);
    HANDLE(CL_INVALID_WORK_ITEM_SIZE);
    HANDLE(CL_MEM_OBJECT_ALLOCATION_FAILURE);
    //HANDLE(CL_MISALIGNED_SUB_BUFFER_OFFSET);
    HANDLE(CL_OUT_OF_RESOURCES);
    HANDLE(CL_OUT_OF_HOST_MEMORY);

    fprintf(stderr, "Unknown OpenCL Error: %d\n", e);
    abort();
}

uint64_t time_ns() {
#ifdef __APPLE__
    uint64_t time = mach_absolute_time();
    mach_timebase_info_data_t info = {0, 0};
    if (info.denom == 0) {
        mach_timebase_info(&info);
    }
    uint64_t time_nano = time * (info.numer / info.denom);
    return time_nano;
#elif __WIN32__
    uint64_t ticks;
    QueryPerformanceCounter((LARGE_INTEGER *)&ticks);
    return ((ticks * ns_per_s) / _ticks_per_s);
#else
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1e9 + ts.tv_nsec);
#endif
}
