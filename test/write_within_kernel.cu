#include <stdio.h> // for CUDA's printf
#include <cstddef> // for CUDA's printf

#include <strf.hpp>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors
 * are of the same length.
 */
__global__ void a_test_kernel()
{
  char buf[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  int global_thread_id = threadIdx.x + blockIdx.x * blockDim.x;
//  strf::snprintf(buf, "Thread %d says: Hello %s\n", global_thread_id, "world.");
//  printf("Thread %3d says: Hello %s\n", global_thread_id, "world.");
  strf::basic_cstr_writer<char> sw(buf);
  write(sw, "Hello");
  write(sw, " World.");
  auto result = sw.finish();

  if (not result.truncated) {
	  printf("Thread %03d's finalized string: \"%s\"\n",  global_thread_id, buf);
  }
  else {
	  printf("Thread %03d's truncated string: \"%26s\"\n", global_thread_id, buf);
  }

//  BOOST_TEST(r.truncated);
//  BOOST_TEST_EQ(*r.ptr, '\0');
//  BOOST_TEST_EQ(r.ptr, &buf[7]);
//  BOOST_TEST_CSTR_EQ(buf, "Hello W");
//
//  printf("%s\n", buf);
}

int main(void)
{
    int threads_per_block = 3;
    int blocks_in_grid = 4;
//    printf("Launching the CUDA kernel on a grid with %d blocks of %d threads each.\n", blocks_in_grid, threads_per_block);
    a_test_kernel<<<blocks_in_grid, threads_per_block>>>();

    cudaDeviceSynchronize();

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    return EXIT_SUCCESS;
}

