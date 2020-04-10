//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <strf.hpp>
#include <iostream>
#include <sstream>
#include <climits>
#include "fmt/format.h"
#include <benchmark/benchmark.h>

#define CREATE_BENCHMARK(PREFIX)                          \
    static void PREFIX (benchmark::State& state) {        \
        for(auto _ : state) {                             \
            auto str = PREFIX ## _OP ;                    \
            benchmark::DoNotOptimize(str.data());         \
            benchmark::ClobberMemory();                   \
        }                                                 \
    }

#define REGISTER_BENCHMARK(X) benchmark::RegisterBenchmark(STR(X ## _OP), X);

#define STR2(X) #X
#define STR(X) STR2(X)

std::string u8sample1(500, 'A');
std::u16string u16sample1(500, u'A');

#define STRF_RC_HELLO_OP       strf::to_string.reserve_calc()("Hello ", "World", "!")
#define STRF_R_HELLO_OP        strf::to_string.reserve(12)("Hello ", "World", "!")
#define STRF_NR_HELLO_OP       strf::to_string.no_reserve()("Hello ", "World", "!")
#define STRF_RC_TR_HELLO_OP    strf::to_string.reserve_calc().tr("Hello {}!", "World")
#define STRF_R_TR_HELLO_OP     strf::to_string.reserve(12).tr("Hello {}!", "World")
#define STRF_NR_TR_HELLO_OP    strf::to_string.no_reserve().tr("Hello {}!", "World")
#define FMT_HELLO_OP           fmt::format("Hello {}!", "World")
#define STRF_RC_I_OP           strf::to_string.reserve_calc()(25)
#define STRF_R_I_OP            strf::to_string.reserve(2)(25)
#define STRF_NR_I_OP           strf::to_string.no_reserve()(25)
#define FMT_I_OP               fmt::format("{}", 25)
#define STD_I_OP               std::to_string(25)
#define STRF_RC_LL_OP          strf::to_string.reserve_calc()(LLONG_MAX)
#define STRF_R_LL_OP           strf::to_string.reserve(20) (LLONG_MAX)
#define STRF_NR_LL_OP          strf::to_string.no_reserve() (LLONG_MAX)
#define FMT_LL_OP              fmt::format("{}", LLONG_MAX)
#define STD_LL_OP              std::to_string(LLONG_MAX)
#define STRF_RC_FILL_OP        strf::to_string .reserve_calc() (strf::right("aa", 20))
#define STRF_R_FILL_OP         strf::to_string .reserve(20) (strf::right("aa", 20))
#define STRF_NR_FILL_OP        strf::to_string .no_reserve() (strf::right("aa", 20))
#define FMT_FILL_OP            fmt::format("{:20}", "aa")
#define STRF_RC_10_20_OP       strf::to_string .reserve_calc() ("ten = ", 10, ", twenty = ", 20)
#define STRF_R_10_20_OP        strf::to_string .reserve(21) ("ten = ", 10, ", twenty = ", 20)
#define STRF_NR_10_20_OP       strf::to_string .no_reserve() ("ten = ", 10, ", twenty = ", 20)
#define STRF_RC_TR_10_20_OP    strf::to_string .reserve_calc() .tr("ten = {}, twenty = {}", 10, 20)
#define STRF_R_TR_10_20_OP     strf::to_string .reserve(21).tr("ten = {}, twenty = {}", 10, 20)
#define STRF_NR_TR_10_20_OP    strf::to_string .no_reserve() .tr("ten = {}, twenty = {}", 10, 20)
#define FMT_10_20_OP           fmt::format("ten = {}, twenty = {}", 10, 20)
#define U8_TO_U16_NR_OP        strf::to_u16string.no_reserve()(strf::cv(u8sample1))
#define U8_TO_U16_RC_OP        strf::to_u16string.reserve_calc()(strf::cv(u8sample1))
#define U8_TO_U16_R_OP         strf::to_u16string.reserve(510)(strf::cv(u8sample1))
#define U16_TO_U8_NR_OP        strf::to_string(strf::cv(u16sample1))
#define U16_TO_U8_RC_OP        strf::to_string.reserve_calc()(strf::cv(u16sample1))
#define U16_TO_U8_R_OP         strf::to_string.reserve(510)(strf::cv(u16sample1))

CREATE_BENCHMARK(STRF_RC_HELLO);
CREATE_BENCHMARK(STRF_R_HELLO);
CREATE_BENCHMARK(STRF_NR_HELLO);
CREATE_BENCHMARK(STRF_RC_TR_HELLO);
CREATE_BENCHMARK(STRF_R_TR_HELLO);
CREATE_BENCHMARK(STRF_NR_TR_HELLO);
CREATE_BENCHMARK(FMT_HELLO);
CREATE_BENCHMARK(STRF_RC_I);
CREATE_BENCHMARK(STRF_R_I);
CREATE_BENCHMARK(STRF_NR_I);
CREATE_BENCHMARK(FMT_I);
CREATE_BENCHMARK(STD_I);
CREATE_BENCHMARK(STRF_RC_LL);
CREATE_BENCHMARK(STRF_R_LL);
CREATE_BENCHMARK(STRF_NR_LL);
CREATE_BENCHMARK(FMT_LL);
CREATE_BENCHMARK(STD_LL);
CREATE_BENCHMARK(STRF_RC_FILL);
CREATE_BENCHMARK(STRF_R_FILL);
CREATE_BENCHMARK(STRF_NR_FILL);
CREATE_BENCHMARK(FMT_FILL);
CREATE_BENCHMARK(STRF_RC_10_20);
CREATE_BENCHMARK(STRF_R_10_20);
CREATE_BENCHMARK(STRF_NR_10_20);
CREATE_BENCHMARK(STRF_RC_TR_10_20);
CREATE_BENCHMARK(STRF_R_TR_10_20);
CREATE_BENCHMARK(STRF_NR_TR_10_20);
CREATE_BENCHMARK(FMT_10_20);

CREATE_BENCHMARK(U8_TO_U16_NR);
CREATE_BENCHMARK(U8_TO_U16_RC);
CREATE_BENCHMARK(U8_TO_U16_R);

CREATE_BENCHMARK(U16_TO_U8_NR);
CREATE_BENCHMARK(U16_TO_U8_RC);
CREATE_BENCHMARK(U16_TO_U8_R);


static void sprintf_10_20(benchmark::State& state)
{
    char buff [40];
    for(auto _ : state) {
        sprintf(buff, "ten = %d, twenty = %d", 10, 20);
        std::string str{buff};
        benchmark::DoNotOptimize(str.data());
        benchmark::ClobberMemory();
    }
}

static void u8_to_u16_buff(benchmark::State& state)
{
    char16_t u16buff [1024];
    for(auto _ : state) {
        (void)strf::to(u16buff)(strf::cv(u8sample1));
        auto str = strf::to_u16string.reserve_calc() (u16buff);
        benchmark::DoNotOptimize(str.data());
        benchmark::ClobberMemory();
    }
}

static void u16_to_u8_buff(benchmark::State& state)
{
    char buff[2000];
    for(auto _ : state) {
        (void)strf::to(buff)(strf::cv(u16sample1));
        auto str = strf::to_string.reserve_calc()(buff);
        benchmark::DoNotOptimize(str.data());
        benchmark::ClobberMemory();
    }
}

int main(int argc, char** argv)
{
    REGISTER_BENCHMARK(STRF_RC_HELLO);
    REGISTER_BENCHMARK(STRF_R_HELLO);
    REGISTER_BENCHMARK(STRF_NR_HELLO);
    REGISTER_BENCHMARK(STRF_RC_TR_HELLO);
    REGISTER_BENCHMARK(STRF_R_TR_HELLO);
    REGISTER_BENCHMARK(STRF_NR_TR_HELLO);
    REGISTER_BENCHMARK(FMT_HELLO);
    REGISTER_BENCHMARK(STRF_RC_I);
    REGISTER_BENCHMARK(STRF_R_I);
    REGISTER_BENCHMARK(STRF_NR_I);
    REGISTER_BENCHMARK(FMT_I);
    REGISTER_BENCHMARK(STD_I);
    REGISTER_BENCHMARK(STRF_RC_LL);
    REGISTER_BENCHMARK(STRF_R_LL);
    REGISTER_BENCHMARK(STRF_NR_LL);
    REGISTER_BENCHMARK(FMT_LL);
    REGISTER_BENCHMARK(STD_LL);
    REGISTER_BENCHMARK(STRF_RC_FILL);
    REGISTER_BENCHMARK(STRF_R_FILL);
    REGISTER_BENCHMARK(STRF_NR_FILL);
    REGISTER_BENCHMARK(FMT_FILL);
    REGISTER_BENCHMARK(STRF_RC_10_20);
    REGISTER_BENCHMARK(STRF_R_10_20);
    REGISTER_BENCHMARK(STRF_NR_10_20);
    REGISTER_BENCHMARK(STRF_RC_TR_10_20);
    REGISTER_BENCHMARK(STRF_R_TR_10_20);
    REGISTER_BENCHMARK(STRF_NR_TR_10_20);
    REGISTER_BENCHMARK(FMT_10_20);
    benchmark::RegisterBenchmark
        ( "sprintf(buff, \"ten = %d, twenty = %d\", 10, 20)"
        , sprintf_10_20 );
    REGISTER_BENCHMARK(U8_TO_U16_NR);
    REGISTER_BENCHMARK(U8_TO_U16_RC);
    REGISTER_BENCHMARK(U8_TO_U16_R);
    benchmark::RegisterBenchmark
        ( "strf::to(buff)(strf::cv(u8sample1)); strf::to_u16string.reserve_calc()(buff)"
        , u8_to_u16_buff );
    REGISTER_BENCHMARK(U16_TO_U8_NR);
    REGISTER_BENCHMARK(U16_TO_U8_RC);
    REGISTER_BENCHMARK(U16_TO_U8_R);
    benchmark::RegisterBenchmark
        ( "strf::to(buff)(strf::cv(u16sample1)); strf::to_string.reserve_calc()(buff)"
        , u16_to_u8_buff );

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}

