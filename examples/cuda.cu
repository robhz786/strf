#include <strf/to_cfile.hpp>


void __device__ device_sample1(strf::destination<char>& out)
{
    strf::to(out) ("At device_sample1.\n");
}

void __device__ device_sample2(strf::destination<char>& out)
{
    strf::to(out) ("At device_sample2.\n");
}

void __global__ kernel_sample(char* dest, std::size_t dest_size, std::size_t* dest_len)
{
    strf::cstr_destination out(dest, dest_size);

    strf::to(out) ("At kernel_sample.\n");
    device_sample1(out);
    strf::to(out) ("After device_sample1.\n");
    device_sample2(out);
    strf::to(out) ("After device_sample2.\n");

    auto result = out.finish();
    *dest_len = result.ptr - dest;
}

int main() {
    struct args {
        char buffer[500];
        std::size_t count;
    };
    args* device_side_args;
    cudaMalloc(&device_side_args, sizeof(args));
    cudaMemset(device_side_args, 0, sizeof(args));
    kernel_sample<<<1, 1>>>( &(device_side_args->buffer[0])
                           , sizeof(device_side_args->buffer)
                           , &(device_side_args->count) );
    cudaDeviceSynchronize();
    args host_side_args = {0};
    cudaMemcpy
        ( &host_side_args
        , device_side_args
        , sizeof(struct args)
        , cudaMemcpyDeviceToHost );
    cudaFree(device_side_args);

    strf::to(stdout) (host_side_args.buffer);
    return 0;
}
