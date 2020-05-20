//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/to_cfile.hpp>
#include <benchmark/benchmark.h>

#define CREATE_BENCHMARK(PREFIX)                             \
    static void PREFIX ## _func (benchmark::State& state) {  \
        const int arr[] = {1, 2, 3, 4, 5, 6};          \
        constexpr int arr_size = sizeof(arr)     \
                                     / sizeof(arr[0]); \
        char dest[110];                                      \
        char* dest_it = dest;                                \
        char* const dest_end = dest + sizeof(dest);          \
        (void) dest_it;                                      \
        (void) dest_end;                                     \
        (void) arr_size;                               \
        for(auto _ : state) {                                \
            PREFIX ## _OP ;                                  \
            benchmark::DoNotOptimize(dest);                  \
            benchmark::ClobberMemory();                      \
        }                                                    \
    }

#define REGISTER_BENCHMARK(X) benchmark::RegisterBenchmark(STR(X ## _OP), X ## _func);
#define REGISTER_BENCHMARK_N(X, NAME) benchmark::RegisterBenchmark(NAME, X ## _func);
#define STR2(X) #X
#define STR(X) STR2(X)

#define RANGE_OP           strf::to(dest) (strf::range(arr));
#define ARGS_OP            strf::to(dest) (1, 2, 3, 4, 5, 6);
#define LOOP_OP            for(int x : arr) \
                               dest_it = strf::to(dest_it, dest_end)(x).ptr;
#define FMT_RANGE_OP       strf::to(dest) (+strf::fmt_range(arr));
#define FMT_ARGS_OP        strf::to(dest) ( +strf::dec(1), +strf::dec(2), +strf::dec(3)   \
                                          , +strf::dec(4), +strf::dec(5), +strf::dec(6) );
#define FMT_LO0P_OP        for(int x : arr) \
                               dest_it = strf::to(dest_it, dest_end)(+strf::dec(x)).ptr;
#define SEP_RANGE_OP       strf::to(dest) (strf::separated_range(arr, "; "));
#define SEP_ARGS_OP        strf::to(dest) (1, "; ", 2, "; ", 3, "; ", 4, "; ", 5, "; ", 6);
#define SEP_LOOP_OP        dest_it = strf::to(dest) (+strf::dec(arr[0])).ptr;                \
                           for (int i = 1; i < arr_size; ++i)                                \
                               dest_it = strf::to(dest_it, dest_end) ("; ", +strf::dec(arr[i])).ptr;
#define FMT_SEP_RANGE_OP   strf::to(dest) (+strf::fmt_separated_range(arr, "; "));
#define FMT_SEP_ARGS_OP    strf::to(dest) ( +strf::dec(1), "; ", +strf::dec(2), "; ", +strf::dec(3) \
                                          , +strf::dec(4), "; ", +strf::dec(5), "; ", +strf::dec(6) );
#define FMT_SEP_LOOP_OP    dest_it = strf::to(dest) (+strf::dec(arr[0])).ptr;  \
                           for (int i = 1; i < arr_size; ++i) \
                               dest_it = strf::to(dest_it, dest_end) ("; ", +strf::dec(arr[i])).ptr;
CREATE_BENCHMARK(RANGE);
CREATE_BENCHMARK(ARGS);
CREATE_BENCHMARK(LOOP);
CREATE_BENCHMARK(FMT_RANGE);
CREATE_BENCHMARK(FMT_ARGS);
CREATE_BENCHMARK(FMT_LO0P);
CREATE_BENCHMARK(SEP_RANGE);
CREATE_BENCHMARK(SEP_ARGS);
CREATE_BENCHMARK(SEP_LOOP);
CREATE_BENCHMARK(FMT_SEP_RANGE);
CREATE_BENCHMARK(FMT_SEP_ARGS);
CREATE_BENCHMARK(FMT_SEP_LOOP);

int main(int argc, char** argv)
{
    REGISTER_BENCHMARK  (RANGE);
    REGISTER_BENCHMARK  (ARGS);
    REGISTER_BENCHMARK  (LOOP);
    REGISTER_BENCHMARK  (FMT_RANGE);
    REGISTER_BENCHMARK_N(FMT_ARGS, "strf::to(dest) (+strf::dec(1), ..., +strf::dec(6)");
    REGISTER_BENCHMARK_N(FMT_LO0P, "/*loop*/");
    REGISTER_BENCHMARK  (SEP_RANGE);
    REGISTER_BENCHMARK  (SEP_ARGS);
    REGISTER_BENCHMARK_N(SEP_LOOP, "/*loop*/");
    REGISTER_BENCHMARK  (FMT_SEP_RANGE);
    REGISTER_BENCHMARK_N(FMT_SEP_ARGS, "strf::to(dest) (+strf::dec(1), \"; \", ..., +strf::dec(6)");
    REGISTER_BENCHMARK_N(FMT_SEP_LOOP, "/*loop*/");

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
