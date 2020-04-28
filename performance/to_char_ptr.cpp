//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <strf.hpp>
#include <iostream>
#include <clocale>
#include <stdio.h>
#include <cstring>
#include <climits>
#include "fmt/format.h"
#include <benchmark/benchmark.h>

#define CREATE_BENCHMARK(PREFIX)                             \
    static void PREFIX ## _func (benchmark::State& state) {  \
        char dest[110];                                       \
        for(auto _ : state) {                                \
            PREFIX ## _OP ;                                  \
            benchmark::DoNotOptimize(dest);                  \
            benchmark::ClobberMemory();                      \
        }                                                    \
    }

#define REGISTER_BENCHMARK(X) benchmark::RegisterBenchmark(STR(X ## _OP), X ## _func);

#define STR2(X) #X
#define STR(X) STR2(X)

#define STRF_HELLO_OP         strf::to(dest)("Hello World!")
#define FMT_HELLO_OP          fmt::format_to(dest, "{}", "Hello World!")
#define STRCPY_HELLO_OP       std::strcpy(dest, "Hello World!")
#define STRF_HELLO2_OP        strf::to(dest)("Hello ", "World", '!')
#define STRF_HELLO3_OP        strf::to(dest)("Hello ", "World", "!")
#define FMT_HELLO2_OP         fmt::format_to(dest, "Hello {}!", "World")
#define SPRINTF_HELLO_OP      std::sprintf(dest, "Hello %s!", "World")
#define STRF_HELLO_LONG_OP    strf::to(dest)("Hello ", long_string, "!")
#define FMT_HELLO_LONG_OP     fmt::format_to(dest, "Hello {}!", long_string)
#define SPRINTF_HELLO_LONG_OP std::sprintf(dest, "Hello %s!", long_string)
#define STRF_FILL_OP          strf::to(dest)(strf::right("aa", 20))
#define STRF_FILL2_OP         strf::to(dest)(strf::join_right(20)("aa"))
#define FMT_FILL_OP           fmt::format_to(dest, "{:20}", "aa")
#define SPRINTF_FILL_OP       std::sprintf(dest, "%20s", "aa")
#define STRF_MIX_OP           strf::to(dest)("blah blah ", INT_MAX, " blah ", *strf::hex(1234)<8, " blah ", "abcdef")
#define STRF_TR_MIX_OP        strf::to(dest).tr("blah blah {} blah {} blah {}", INT_MAX, *strf::hex(1234)<8, "abcdef")
#define FMT_MIX_OP            fmt::format_to(dest, "blah blah {} blah {:<#8x} blah {}", INT_MAX, 1234, "abcdef")
#define SPRINTF_MIX_OP        std::sprintf(dest, "blah blah %d blah %#-8x blah %s", INT_MAX, 1234, "abcdef")
#define STRF_10_20_OP         strf::to(dest)("ten =  ", 10, ", twenty = ", 20)
#define STRF_TR_10_20_OP      strf::to(dest).tr("ten = {}, twenty = {}", 10, 20)
#define FMT_10_20_OP          fmt::format_to(dest, "ten = {}, twenty = {}", 10, 20)
#define SPRINTF_10_20_OP      std::sprintf(dest, "ten = %d, twenty= %d", 10, 20)

CREATE_BENCHMARK(STRF_HELLO);
CREATE_BENCHMARK(FMT_HELLO);
CREATE_BENCHMARK(STRCPY_HELLO);
CREATE_BENCHMARK(STRF_HELLO2);
CREATE_BENCHMARK(STRF_HELLO3);
CREATE_BENCHMARK(FMT_HELLO2);
CREATE_BENCHMARK(SPRINTF_HELLO);
CREATE_BENCHMARK(STRF_FILL);
CREATE_BENCHMARK(STRF_FILL2);
CREATE_BENCHMARK(FMT_FILL);
CREATE_BENCHMARK(SPRINTF_FILL);
CREATE_BENCHMARK(STRF_MIX);
CREATE_BENCHMARK(STRF_TR_MIX);
CREATE_BENCHMARK(FMT_MIX);
CREATE_BENCHMARK(SPRINTF_MIX);
CREATE_BENCHMARK(STRF_10_20);
CREATE_BENCHMARK(STRF_TR_10_20);
CREATE_BENCHMARK(FMT_10_20);
CREATE_BENCHMARK(SPRINTF_10_20);


int main(int argc, char** argv)
{
    REGISTER_BENCHMARK(STRF_HELLO);
    REGISTER_BENCHMARK(FMT_HELLO);
    REGISTER_BENCHMARK(STRCPY_HELLO);
    REGISTER_BENCHMARK(STRF_HELLO2);
    REGISTER_BENCHMARK(STRF_HELLO3);
    REGISTER_BENCHMARK(FMT_HELLO2);
    REGISTER_BENCHMARK(SPRINTF_HELLO);
    REGISTER_BENCHMARK(STRF_FILL);
    REGISTER_BENCHMARK(STRF_FILL2);
    REGISTER_BENCHMARK(FMT_FILL);
    REGISTER_BENCHMARK(SPRINTF_FILL);
    REGISTER_BENCHMARK(STRF_MIX);
    REGISTER_BENCHMARK(STRF_TR_MIX);
    REGISTER_BENCHMARK(FMT_MIX);
    REGISTER_BENCHMARK(SPRINTF_MIX);
    REGISTER_BENCHMARK(STRF_10_20);
    REGISTER_BENCHMARK(STRF_TR_10_20);
    REGISTER_BENCHMARK(FMT_10_20);
    REGISTER_BENCHMARK(SPRINTF_10_20);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
