#include <stdio.h> // for CUDA's printf

#include "test_utils.hpp"
#include <strf.hpp>
#include <sstream>
#include <iostream>

namespace kernels {

// Note: There are adaptations of, say, std::span for use with CUDA (= I adapted it...).
// But we want to avoid dependency clutter here, so let's just stick to the basics.

__global__ void using_cstr_writer(strf::cstr_writer::result* write_result, char* buffer, std::size_t buffer_size)
{
	strf::basic_cstr_writer<char> sw(buffer, buffer_size);
	write(sw, "Hello");
	write(sw, " world");
	*write_result = sw.finish();
}

__global__ void using_cstr_to(char* buffer, std::size_t buffer_size)
{
	int global_thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	auto printer = strf::to(buffer, buffer_size);
	printer ( "Hello", ' ', "world, from thread ", global_thread_id);
}

__global__ void various_types_with_cstr_writer(char* buffer, std::size_t buffer_size)
{
	strf::cstr_writer writer(buffer, buffer_size);
	auto printer = strf::to(writer);

	printer("Printing a bool: ", bool(true), '\n');
//	printer("Printing a wchar_t: ", wchar_t('a'), '\n');
//	printer("Printing a char16_t: ", char16_t('a'), '\n');
//	printer("Printing a char32_t: ", char32_t('a'), '\n');
	printer("Printing a char: ", static_cast<char>('a'), '\n');
	printer("Printing a unsigned char: ", static_cast<unsigned char>('a'), '\n');
	printer("Printing a short: ", short(-1234), '\n');
	printer("Printing a unsigned short: ", static_cast<unsigned short>(1234), '\n');
	printer("Printing a int: ", int(-12345678), '\n');
	printer("Printing a signed int: ", static_cast<signed int>(12345678), '\n');
	printer("Printing a unsigned int: ", static_cast<unsigned int>(12345678), '\n');
	printer("Printing a long int: ", static_cast<long int>(-12345678900), '\n');
	printer("Printing a signed long int: ", static_cast<signed long int>(-12345678900), '\n');
	printer("Printing a unsigned long int: ", static_cast<unsigned long int>(12345678900), '\n');
	printer("Printing a long long int: ", static_cast<long long int>(-12345678900), '\n');
	printer("Printing a signed long long int: ", static_cast<signed long long int>(-12345678900), '\n');
	printer("Printing a unsigned long long int: ", static_cast<unsigned long long int>(12345678900), '\n');
	printer("Printing a float: ", float(1.234567), '\n');
	printer("Printing a double: ", double(1.2345678901234567), '\n');
	writer.finish();
	printf("Buffer contents: \"%s\"", buffer);
}

__global__ void formatting_functions(char* buffer, std::size_t buffer_size)
{
	strf::cstr_writer writer(buffer, buffer_size);
	auto printer = strf::to(writer);

	printer ("strf::fmt(0) gives ", strf::fmt(0), '\n');
	printer ("strf::fmt(0).hex() gives ", strf::fmt(0).hex(), '\n');
	printer ("strf::fmt(0).bin() gives ", strf::fmt(0).bin(), '\n');
	printer ("strf::left(0, 2, '0') gives ", strf::left(0, 2, '0'), '\n');
	printer ("strf::right(0, 2, '0') gives ", strf::right(0, 2, '0'), '\n');
	printer ("strf::fmt(123) gives ", strf::fmt(123), '\n');
	printer ("strf::fmt(123).hex() gives ", strf::fmt(123).hex(), '\n');
	printer ("strf::fmt(123).bin() gives ", strf::fmt(123).bin(), '\n');
	printer ("strf::left(123, 5, '0') gives ", strf::left(123, 5, '0'), '\n');
	printer ("strf::right(123, 5, '0') gives ", strf::right(123, 5, '0'), '\n');

	writer.finish();
}

} // namespace kernels

// Ugly, no-good error-checking.
#define ensure_cuda_success(ans) { ensure_cuda_success_((ans), __FILE__, __LINE__); }

inline void ensure_cuda_success_(cudaError_t status, const char *file, int line, bool abort=true)
{
	TEST_EQ(status, cudaSuccess);
	if (abort and (status != cudaSuccess)) {
		TEST_ERROR(cudaGetErrorString(status));
		exit(test_finish());
	}
}

void test_cstr_writer()
{
	struct args {
		strf::cstr_writer::result write_result;
		char buffer[50];
	};
	const std::size_t buffer_size { std::strlen("Hello world") + 1 }; // Enough for "Hello world" with the trailing '\0'.
	struct args* device_side_args;
	ensure_cuda_success(cudaMalloc(&device_side_args, sizeof(struct args)));
	ensure_cuda_success(cudaMemset(device_side_args, 0, sizeof(struct args)));

	int threads_per_block { 1 };
	int blocks_in_grid { 1 };
		// We could theoretically have multiple threads in multiple blocks run this, but
		// it shouldn't really matter.
	kernels::using_cstr_writer<<<threads_per_block, blocks_in_grid>>>(
		&(device_side_args->write_result),
		&(device_side_args->buffer[0]),
		buffer_size);
	ensure_cuda_success(cudaGetLastError());
	ensure_cuda_success(cudaDeviceSynchronize());
	args host_side_args;
	ensure_cuda_success(cudaMemcpy(&host_side_args, device_side_args, sizeof(struct args), cudaMemcpyDeviceToHost));
	TEST_EQ(host_side_args.write_result.truncated, false);
	TEST_EQ(host_side_args.write_result.ptr, &(device_side_args->buffer[0]) + std::strlen("Hello world"));
	if (host_side_args.write_result.ptr == &(device_side_args->buffer[0])) {
		TEST_EQ(strncmp(host_side_args.write_result.ptr, host_side_args.buffer, buffer_size), 0);
	}
}


void test_cstr_to()
{
	char* device_side_buffer;
	const std::size_t buffer_size { 100 }; // More than enough for "Hello world from thread XYZ"
	ensure_cuda_success( cudaMalloc(&device_side_buffer, buffer_size) );
	ensure_cuda_success( cudaMemset(device_side_buffer, 0, buffer_size) );

	int threads_per_block { 1 };
	int blocks_in_grid { 1 };
		// We could theoretically have multiple threads in multiple blocks run this, but
		// it shouldn't really matter.
	kernels::using_cstr_to<<<threads_per_block, blocks_in_grid>>>(device_side_buffer, buffer_size);
	ensure_cuda_success(cudaGetLastError());
	ensure_cuda_success(cudaDeviceSynchronize());
	char host_side_buffer[buffer_size];
	ensure_cuda_success(cudaMemcpy(&host_side_buffer, device_side_buffer, buffer_size , cudaMemcpyDeviceToHost));
	std::stringstream expected;
	expected << "Hello" << ' ' << "world, from thread " << 0;
	TEST_EQ(strncmp(host_side_buffer, expected.str().c_str(), buffer_size), 0);
	std::cout << std::endl;
	std::cout << "Result: \"" << host_side_buffer << "\"\n";
	std::cout << "Expected: \"" << expected.str() <<  "\"\n";
}

void test_various_types_with_cstr_writer()
{
	char* device_side_buffer;
	const std::size_t buffer_size { 5000 }; 
	ensure_cuda_success( cudaMalloc(&device_side_buffer, buffer_size) );
	ensure_cuda_success( cudaMemset(device_side_buffer, 0, buffer_size) );

	int threads_per_block { 1 };
	int blocks_in_grid { 1 };
		// We could theoretically have multiple threads in multiple blocks run this, but
		// it shouldn't really matter.
	kernels::various_types_with_cstr_writer<<<threads_per_block, blocks_in_grid>>>(device_side_buffer, buffer_size);
	ensure_cuda_success(cudaGetLastError());
	ensure_cuda_success(cudaDeviceSynchronize());
	char host_side_buffer[buffer_size];
	ensure_cuda_success(cudaMemcpy(&host_side_buffer, device_side_buffer, buffer_size , cudaMemcpyDeviceToHost));
	std::stringstream expected;
	expected
		<< "Printing a bool: " << bool(true) << '\n'
		<< "Printing a char: " << char('a') << '\n'
		<< "Printing a unsigned char: " << static_cast<unsigned char>('a') << '\n'
		<< "Printing a short: " << short(-1234) << '\n'
		<< "Printing a unsigned short: " << static_cast<unsigned short>(1234) << '\n'
		<< "Printing a int: " << int(-12345678) << '\n'
		<< "Printing a signed int: " << static_cast<signed int>(12345678) << '\n'
		<< "Printing a unsigned int: " << static_cast<unsigned int>(12345678) << '\n'
		<< "Printing a long int: " << static_cast<long int>(-12345678900) << '\n'
		<< "Printing a signed long int: " << static_cast<signed long int>(-12345678900) << '\n'
		<< "Printing a unsigned long int: " << static_cast<unsigned long int>(12345678900) << '\n'
		<< "Printing a long long int: " << static_cast<long long int>(-12345678900) << '\n'
		<< "Printing a signed long long int: " << static_cast<signed long long int>(-12345678900) << '\n'
		<< "Printing a unsigned long long int: " << static_cast<unsigned long long int>(12345678900) << '\n'
		<< "Printing a float: " << float(1.234567) << '\n'
		<< "Printing a double: " << double(1.2345678901234567), '\n';
	TEST_EQ(strncmp(host_side_buffer, expected.str().c_str(), buffer_size), 0);
	std::cout << std::endl;
	std::cout << "Result: \"" << host_side_buffer << "\"\n";
	std::cout << "Expected: \"" << expected.str() <<  "\"\n";
}

void test_formatting_functions()
{
	char* device_side_buffer;
	constexpr std::size_t buffer_size { 400 };
	ensure_cuda_success( cudaMalloc(&device_side_buffer, buffer_size) );
	ensure_cuda_success( cudaMemset(device_side_buffer, 0, buffer_size) );

	int threads_per_block { 1 };
	int blocks_in_grid { 1 };
	kernels::formatting_functions<<<threads_per_block, blocks_in_grid>>>(device_side_buffer, buffer_size);
	ensure_cuda_success(cudaGetLastError());
	ensure_cuda_success(cudaDeviceSynchronize());
	char host_side_buffer[buffer_size];
	ensure_cuda_success(cudaMemcpy(&host_side_buffer, device_side_buffer, buffer_size , cudaMemcpyDeviceToHost));
	std::stringstream expected;
	expected  <<
		"strf::fmt(0) gives 0\n"
		"strf::fmt(0).hex() gives 0\n"
		"strf::fmt(0).bin() gives 0\n"
		"strf::left(0, 2, '0') gives 00\n"
		"strf::right(0, 2, '0') gives 00\n"
		"strf::fmt(123) gives 123\n"
		"strf::fmt(123).hex() gives 7b\n"
		"strf::fmt(123).bin() gives 1111011\n"
		"strf::left(123, 5, '0') gives 12300\n"
		"strf::right(123, 5, '0') gives 00123\n";
	TEST_EQ(strncmp(host_side_buffer, expected.str().c_str(), buffer_size), 0);
	std::cout << std::endl;
	std::cout << "Result: \"" << host_side_buffer << "\"\n";
	std::cout << "Expected: \"" << expected.str() <<  "\"\n";
}


void cstr_to_sanity_check()
{
	const std::size_t buffer_size { 100 }; // More than enough for "Hello world from thread XYZ"
	char buffer[buffer_size];
	std::fill_n(buffer, sizeof(buffer), 0);
	auto print_functor = strf::to(buffer, buffer_size);
	print_functor ( "Hello", ' ', "world, from thread ", 1 );
	std::stringstream expected;
	expected << "Hello" << ' ' << "world, from thread " << 1;
	TEST_EQ(strncmp(buffer, expected.str().c_str(), buffer_size), 0);
}


int main(void)
{
	auto num_devices { 0 };
	auto status = cudaGetDeviceCount(&num_devices);

	TEST_EQ(status, cudaSuccess);
	if (status != cudaSuccess)
	{
		std::stringstream ss;
		ss << "cudaGetDeviceCount failed: " << cudaGetErrorString(status) <<  '\n';
		TEST_ERROR(ss.str().c_str());
	}
	if (num_devices == 0) {
		std::cerr << "No devices - can't run this test\n";
		return test_finish();
	}
	// TODO: Test basic_cstr_writer's with different character types
	test_cstr_writer();
	cstr_to_sanity_check();
	test_cstr_to();
	test_formatting_functions();
	test_various_types_with_cstr_writer();

	cudaDeviceReset();
	return test_finish();
}
