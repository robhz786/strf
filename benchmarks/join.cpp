//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <benchmark/benchmark.h>
#include <strf/to_cfile.hpp>

#define CREATE_BENCHMARK(PREFIX)                          \
    static void PREFIX (benchmark::State& state) {        \
        char dest[100];                                   \
        for(auto _ : state) {                             \
            (void) PREFIX ## _OP ;                        \
            benchmark::DoNotOptimize(dest);               \
            benchmark::ClobberMemory();                   \
        }                                                 \
    }

#define REGISTER_BENCHMARK(X) benchmark::RegisterBenchmark(STR(X ## _OP), X);

#define STR2(X) #X
#define STR(X) STR2(X)

#define BM_JOIN_4_OP   strf::to(dest) (strf::join('a', 'b', 'c', 'd'))
#define BM_4_OP        strf::to(dest) ('a', 'b', 'c', 'd')
#define BM_JOIN_8_OP   strf::to(dest) (strf::join('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'))
#define BM_8_OP        strf::to(dest) ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
#define BM_JR_8_OP     strf::to(dest) (strf::join_right(15)('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'))
#define BM_JS_8_OP     strf::to(dest) (strf::join_split(15, 2)('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'))
#define BM_JR_STR_OP   strf::to(dest) (strf::join_right(15)("Hello World"))
#define BM_JR2_STR_OP  strf::to(dest) (strf::join("Hello World") > 15)
#define BM_STR_OP      strf::to(dest) (strf::fmt("Hello World") > 15)
#define BM_JRI_OP      strf::to(dest) (strf::join_right(4)(25))
#define BM_RI_OP       strf::to(dest) (strf::dec(25) > 4)

CREATE_BENCHMARK(BM_JOIN_4);
CREATE_BENCHMARK(BM_4);
CREATE_BENCHMARK(BM_JOIN_8);
CREATE_BENCHMARK(BM_8);
CREATE_BENCHMARK(BM_JR_8);
CREATE_BENCHMARK(BM_JS_8);
CREATE_BENCHMARK(BM_JR_STR);
CREATE_BENCHMARK(BM_JR2_STR);
CREATE_BENCHMARK(BM_STR);
CREATE_BENCHMARK(BM_JRI);
CREATE_BENCHMARK(BM_RI);

int main(int argc, char** argv)
{
    REGISTER_BENCHMARK(BM_JOIN_4);
    REGISTER_BENCHMARK(BM_4);
    REGISTER_BENCHMARK(BM_JOIN_8);
    REGISTER_BENCHMARK(BM_8);
    REGISTER_BENCHMARK(BM_JR_8);
    REGISTER_BENCHMARK(BM_JS_8);
    REGISTER_BENCHMARK(BM_JR_STR);
    REGISTER_BENCHMARK(BM_JR2_STR);
    REGISTER_BENCHMARK(BM_STR);
    REGISTER_BENCHMARK(BM_JRI);
    REGISTER_BENCHMARK(BM_RI);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}


