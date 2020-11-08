//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.cpp"

#include <strf/to_cfile.hpp>

namespace test_utils {

strf::outbuff*& STRF_HD test_outbuff_ptr()
{
    static strf::outbuff* ptr = nullptr;
    return ptr;
}

void STRF_HD set_test_outbuff(strf::outbuff& ob)
{
    test_outbuff_ptr() = &ob;
}

strf::outbuff& STRF_HD test_outbuff()
{
    auto * ptr = test_outbuff_ptr();
    return *ptr;
}

} // namespace test_utils

extern void __device__ test_cstr_writer();
extern void __device__ test_dynamic_charset();
extern void __device__ test_encode_char();
extern void __device__ test_encode_fill();
extern void __device__ test_facets_pack();
extern void __device__ test_facets_pack_merge();
extern void __device__ test_input_bool();
extern void __device__ test_input_char();
extern void __device__ test_input_char32();
extern void __device__ test_input_facets_pack();
extern void __device__ test_input_float();
extern void __device__ test_input_int();
extern void __device__ test_input_ptr();
extern void __device__ test_input_range();
extern void __device__ test_input_string();
extern void __device__ test_join();
extern void __device__ test_miscellaneous();
extern void __device__ test_numpunct();
extern void __device__ test_printable_overriding();
extern void __device__ test_reserve();
extern void __device__ test_single_byte_encodings();
extern void __device__ test_tr_string();
extern void __device__ test_width_calculator();
extern void __device__ test_width_t();
extern void __device__ test_utf_to_utf();

namespace kernels {

__global__ void kernel_main
    ( unsigned* errors_count
    , char* err_msg
    , std::size_t err_msg_size )
{
    strf::cstr_writer out(err_msg, err_msg_size);
    test_utils::set_test_outbuff(out);

    test_cstr_writer();
    // test_locale();            // not supported on CUDA
    // test_cfile_writer();      // not supported on CUDA
    // test_streambuf_writer();  // not supported on CUDA
    // test_string_writer();     // not supported on CUDA

    // test_dynamic_charset();   // not supported on CUDA
    test_encode_char();
    test_encode_fill();
    test_facets_pack();
    test_facets_pack_merge();
    test_input_bool();
    test_input_char();
    test_input_char32();
    test_input_facets_pack();
    test_input_float();
    test_input_int();
    test_input_ptr();
    test_input_range();
    test_input_string();
    test_printable_overriding();
    test_join();
    test_miscellaneous();
    test_numpunct();
    test_reserve();
    test_single_byte_encodings();
    test_tr_string();
    test_utf_to_utf();
    test_width_calculator();
    test_width_t();

    auto result = out.finish();
    (void)result;
    *errors_count = test_utils::test_err_count();
}

} // namespace kernels


int main() {
    auto print = strf::to(stdout);
    auto num_devices { 0 };
    cudaError_t status = cudaGetDeviceCount(&num_devices);

    if (status != cudaSuccess) {
        print ("cudaGetDeviceCount failed: ", cudaGetErrorString(status), '\n');
        return status;
    }
    if (num_devices == 0) {
        print ("No devices - can't run this test\n");
        return status;
    }

    constexpr std::size_t stackSize = 50 * 1024;
    status = cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
    if (status != cudaSuccess) {
        print( "cudaDeviceSetLimit(cudaLimitStackSize, ", stackSize, ") failed: "
             , cudaGetErrorString(status), '\n');
        cudaDeviceReset();
        return status;
    }

    constexpr std::size_t buffer_size = 2000;
    struct args {
        unsigned errors_count;
        char buffer[buffer_size];
    };
    struct args* device_side_args;
    status = cudaMalloc(&device_side_args, sizeof(struct args));
    if (status != cudaSuccess) {
        print("cudaMalloc failed: ", cudaGetErrorString(status), '\n');
        cudaDeviceReset();
        return status;
    }
    status = cudaMemset(device_side_args, 0, sizeof(struct args));
    if (status != cudaSuccess) {
        print("cudaMemset failed: ", cudaGetErrorString(status), '\n');
        cudaDeviceReset();
        return status;
    }

    int threads_per_block { 1 };
    int blocks_in_grid { 1 };

    kernels::kernel_main<<<threads_per_block, blocks_in_grid>>>(
        &(device_side_args->errors_count),
        &(device_side_args->buffer[0]),
        buffer_size );
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        print("kernel_main<<<", threads_per_block, ',', blocks_in_grid, ">>> failed: "
             ,cudaGetErrorString(status), '\n' );
        return status;
    }
    status = cudaDeviceSynchronize();
    // if you get "cudaDeviceSynchronize() failed: an illegal memory access was encountered"
    // then try increasing `stackSize` variable above
    if (status != cudaSuccess) {
        print("cudaDeviceSynchronize() failed: ", cudaGetErrorString(status), '\n');
        return status;
    }
    args host_side_args;
    status = cudaMemcpy
        ( &host_side_args, device_side_args
        , sizeof(struct args), cudaMemcpyDeviceToHost );
    if (status != cudaSuccess) {
        print("cudaMemcpy failed: ", cudaGetErrorString(status), '\n');
        cudaDeviceReset();
        return status;
    }
    cudaFree(device_side_args);
    cudaDeviceReset();

    print (host_side_args.buffer);
    if (host_side_args.errors_count == 0) {
        print("All test passed!\n");
    } else {
        print(host_side_args.errors_count, " tests failed!\n");
    }
    std::fflush(stdout);

    return  host_side_args.errors_count;
}
