//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <strf/to_string.hpp>
#include <strf/to_cfile.hpp>
#include <iostream>
#include <sstream>
#include <climits>
#include "fmt/format.h"
#include "fmt/compile.h"
#include <benchmark/benchmark.h>

#define STR2(X) #X
#define STR(X) STR2(X)
#define CAT2(X, Y) X##Y
#define CAT(X,Y) CAT2(X,Y)
#define BM(FIXTURE, EXPR)  BM2(CAT(bm_, __LINE__) , FIXTURE, EXPR)
#define BM2(ID, FIXTURE, EXPR)                                          \
    struct ID {                                                         \
        static void func(benchmark::State& state) {                     \
            FIXTURE;                                                    \
            for(auto _ : state) {                                       \
                auto s = EXPR ;                                         \
                benchmark::DoNotOptimize(s.data());                     \
                benchmark::ClobberMemory();                             \
            }                                                           \
        }                                                               \
    };                                                                  \
    benchmark::RegisterBenchmark(STR(EXPR), ID :: func);



#define FIXTURE_STR std::string str = "blah blah blah blah blah blah blah";
#define FMT_FIXTURE_STR auto compiled_fmt_str = fmt::compile<std::string>("Blah {}!\n");

#define FMT_FIXTURE_INT  auto compiled_fmt_int = fmt::compile<int>("{}");
#define FMT_FIXTURE_DBL  auto compiled_fmt_double = fmt::compile<double>("{}");
#define FMT_FIXTURE1  auto compiled_fmt1 = (fmt::compile<int, int>("blah {} blah {} blah"));
#define FMT_FIXTURE2  auto compiled_fmt2 = (fmt::compile<int, int>("blah {:+} blah {:#x} blah"));
#define FMT_FIXTURE3  auto compiled_fmt3 = (fmt::compile<int, int>("blah {:_>+20} blah {:<#20x} blah"));

#define FIXTURE_U8SAMPLE  auto u8sample = std::string(500, 'A');
#define FIXTURE_U16SAMPLE auto u16sample = std::string(500, 'A');

// static void u8_to_u16_buff(benchmark::State& state)
// {
//     char16_t u16buff [510];
//     FIXTURE_U8SAMPLE;
//     for(auto _ : state) {
//         (void)strf::to(u16buff)(strf::conv(u8sample));
//         auto str = strf::to_u16string.reserve_calc() (u16buff);
//         benchmark::DoNotOptimize(str.data());
//         benchmark::ClobberMemory();
//     }
// }

// static void u16_to_u8_buff(benchmark::State& state)
// {
//     char buff[510];
//     FIXTURE_U16SAMPLE;
//     for(auto _ : state) {
//         (void)strf::to(buff)(strf::conv(u16sample));
//         auto str = strf::to_string.reserve_calc()(buff);
//         benchmark::DoNotOptimize(str.data());
//         benchmark::ClobberMemory();
//     }
// }
constexpr auto to_string_rc = strf::to_string.reserve_calc();
constexpr auto to_string_nr = strf::to_string.no_reserve();

int main(int argc, char** argv)
{
    BM(, std::to_string(123456));
    BM(, std::to_string(0.123456));

    BM(, to_string_rc(123456));
    BM(, to_string_rc(0.123456));
    BM(FIXTURE_STR, to_string_rc("Blah ", str, "!\n"));
    BM(, to_string_rc("blah ", 123456, " blah ", 0x123456, " blah"));
    BM(, to_string_rc("blah ", +strf::dec(123456), " blah ", *strf::hex(0x123456), " blah"));
    BM(, to_string_rc("blah ", +strf::right(123456, 20, '_'), " blah ", *strf::hex(0x123456)<20, " blah"));

    BM(,to_string_nr(123456));
    BM(,to_string_nr(0.123456));
    BM(FIXTURE_STR, to_string_nr("Blah ", str, "!\n"));
    BM(, to_string_nr("blah ", 123456, " blah ", 0x123456, " blah"));
    BM(, to_string_nr("blah ", +strf::dec(123456), " blah ", *strf::hex(0x123456), " blah"));
    BM(, to_string_nr("blah ", +strf::right(123456, 20, '_'), " blah ", *strf::hex(0x123456)<20, " blah"));

    BM(, to_string_rc.tr("{}", 123456));
    BM(, to_string_rc.tr("{}", 0.123456));
    BM(FIXTURE_STR, to_string_rc.tr("Blah {}!\n", str));
    BM(, to_string_rc.tr("blah {} blah {} blah", 123456, 0x123456));
    BM(, to_string_rc.tr("blah {} blah {} blah", +strf::dec(123456), *strf::hex(0x123456)));
    BM(, to_string_rc.tr("blah {} blah {} blah", +strf::right(123456, 20, '_'), *strf::hex(0x123456)<20));

    BM(, to_string_nr.tr("{}", 123456));
    BM(, to_string_nr.tr("{}", 0.123456));
    BM(FIXTURE_STR, to_string_nr.tr("Blah {}!\n", str));
    BM(, to_string_nr.tr("blah {} blah {} blah", 123456, 0x123456));
    BM(, to_string_nr.tr("blah {} blah {} blah", +strf::dec(123456), *strf::hex(0x123456)));
    BM(, to_string_nr.tr("blah {} blah {} blah", +strf::right(123456, 20, '_'), *strf::hex(0x123456)<20));

    BM(FMT_FIXTURE_INT, fmt::format(compiled_fmt_int, 123456));
    BM(FMT_FIXTURE_DBL, fmt::format(compiled_fmt_double, 0.123456));
    BM(FIXTURE_STR; FMT_FIXTURE_STR, fmt::format(compiled_fmt_str, str));
    BM(FMT_FIXTURE1, fmt::format(compiled_fmt1, 123456, 0x123456));
    BM(FMT_FIXTURE2, fmt::format(compiled_fmt2, 123456, 0x123456));
    BM(FMT_FIXTURE3, fmt::format(compiled_fmt3, 123456, 0x123456));

    BM(, fmt::format("{}", 123456));
    BM(, fmt::format("{}", 0.123456));
    BM(FIXTURE_STR, fmt::format("Blah {}!\n", str));
    BM(, fmt::format("blah {} blah {} blah", 123456, 0x123456));
    BM(, fmt::format("blah {:+} blah {:#x} blah", 123456, 0x123456));
    BM(, fmt::format("blah {:_>+20} blah {:<#20x} blah", 123456, 0x123456));

    BM(FIXTURE_U8SAMPLE, strf::to_u16string.reserve_calc() (strf::conv(u8sample)));
    BM(FIXTURE_U8SAMPLE, strf::to_u16string.no_reserve()   (strf::conv(u8sample)));
    BM(FIXTURE_U8SAMPLE, strf::to_u16string.reserve(510)   (strf::conv(u8sample)));
    // benchmark::RegisterBenchmark
    //     ( "strf::to(u16buff)(strf::conv(u8sample)); strf::to_u16string.reserve_calc()(u16buff)"
    //     , u8_to_u16_buff );

    BM(FIXTURE_U16SAMPLE, strf::to_string.reserve_calc() (strf::conv(u16sample)));
    BM(FIXTURE_U16SAMPLE, strf::to_string.no_reserve()   (strf::conv(u16sample)));
    BM(FIXTURE_U16SAMPLE, strf::to_string.reserve(510)   (strf::conv(u16sample)));
    // benchmark::RegisterBenchmark
    //     ( "strf::to(u8buff)(strf::conv(u16sample)); to_string_rc(u8buff)"
    //     , u16_to_u8_buff );

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    strf::to(stdout)
        ( "\n  where :"
          "\n    constexpr auto to_string_rc = strf::to_string.reserve_calc();"
          "\n    constexpr auto to_string_nr = strf::to_string.no_reserve();"
          "\n    " STR(FIXTURE_STR)
          "\n    " STR(FMT_FIXTURE_STR)
          "\n    " STR(FMT_FIXTURE1)
          "\n    " STR(FMT_FIXTURE2)
          "\n    " STR(FMT_FIXTURE3)
          "\n    " STR(FIXTURE_U8SAMPLE)
          "\n    " STR(FIXTURE_U16SAMPLE)
          "\n" );

    return 0;
}

