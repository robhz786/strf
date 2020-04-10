//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>
#include <benchmark/benchmark.h>
#include "fmt/format.h"

#define CREATE_BENCHMARK(PREFIX)                             \
    static void PREFIX ## _func (benchmark::State& state) {  \
        char dest[110];                                      \
        for(auto _ : state) {                                \
            PREFIX ## _OP ;                                  \
            benchmark::DoNotOptimize(dest);                  \
            benchmark::ClobberMemory();                      \
        }                                                    \
    }

#define REGISTER_BENCHMARK(X) benchmark::RegisterBenchmark(STR(X ## _OP), X ## _func);
#define STR2(X) #X
#define STR(X) STR2(X)

#define STRF_HELLO_OP strf::to(dest) .tr("{}", "Hello World!");
#define STRF_10_20_OP strf::to(dest).tr("ten = {}, twenty = {}", 10, 20);
#define STRF_POS_OP strf::to(dest).tr("ten = {1}, twenty = {0}", 20, 10);

#define FMT_HELLO_OP fmt::format_to(dest, "{}", "Hello World!");
#define FMT_10_20_OP fmt::format_to(dest, "ten = {}, twenty = {}", 10, 20);
#define FMT_POS_OP fmt::format_to(dest, "ten = {1}, twenty = {0}", 20, 10);

CREATE_BENCHMARK(STRF_HELLO);
CREATE_BENCHMARK(STRF_10_20);
CREATE_BENCHMARK(STRF_POS);
CREATE_BENCHMARK(FMT_HELLO);
CREATE_BENCHMARK(FMT_10_20);
CREATE_BENCHMARK(FMT_POS);

int main(int argc, char** argv)
{
    REGISTER_BENCHMARK(STRF_HELLO);
    REGISTER_BENCHMARK(STRF_10_20);
    REGISTER_BENCHMARK(STRF_POS);
    REGISTER_BENCHMARK(FMT_HELLO);
    REGISTER_BENCHMARK(FMT_10_20);
    REGISTER_BENCHMARK(FMT_POS);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
