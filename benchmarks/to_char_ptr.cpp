//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <strf/to_cfile.hpp>
#include <stdio.h>
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
            char dest[110];                                             \
            for(auto _ : state) {                                       \
                EXPR ;                                                  \
                benchmark::DoNotOptimize(dest);                         \
                benchmark::ClobberMemory();                             \
            }                                                           \
        }                                                               \
    };                                                                  \
    benchmark::RegisterBenchmark(STR(EXPR), ID :: func);

#define FIXTURE_STR std::string str = "blah blah blah blah blah blah blah";
#define FMT_FIXTURE_STR auto compiled_fmt_str = fmt::compile<std::string>("Blah {}!\n");
#define FMT_FIXTURE1  auto compiled_fmt1 = (fmt::compile<int, int>("blah {} blah {} blah"));
#define FMT_FIXTURE2  auto compiled_fmt2 = (fmt::compile<int, int>("blah {:+} blah {:#x} blah"));
#define FMT_FIXTURE3  auto compiled_fmt3 = (fmt::compile<int, int>("blah {:_>+20} blah {:<#20x} blah"));

int main(int argc, char** argv)
{
    BM(FIXTURE_STR, strf::to(dest)("Blah ", str, "!\n"));
    BM(, strf::to(dest)("blah ", 123456, " blah ", 0x123456, " blah"));
    BM(, strf::to(dest)("blah ", +strf::dec(123456), " blah ", *strf::hex(0x123456), " blah"));
    BM(, strf::to(dest)("blah ", +strf::right(123456, 20, '_'), " blah ", *strf::hex(0x123456)<20, " blah"));

    BM(FIXTURE_STR, strf::to(dest).tr("Blah {}!\n", str));
    BM(, strf::to(dest).tr("blah {} blah {} blah", 123456, 0x123456));
    BM(, strf::to(dest).tr("blah {} blah {} blah", +strf::dec(123456), *strf::hex(0x123456)));
    BM(, strf::to(dest).tr("blah {} blah {} blah", +strf::right(123456, 20, '_'), *strf::hex(0x123456)<20));

    BM(FIXTURE_STR; FMT_FIXTURE_STR, fmt::format_to(dest, compiled_fmt_str, str));
    BM(FMT_FIXTURE1, fmt::format_to(dest, compiled_fmt1, 123456, 0x123456));
    BM(FMT_FIXTURE2, fmt::format_to(dest, compiled_fmt2, 123456, 0x123456));
    BM(FMT_FIXTURE3, fmt::format_to(dest, compiled_fmt3, 123456, 0x123456));

    BM(FIXTURE_STR, fmt::format_to(dest, "Blah {}!\n", str));
    BM(, fmt::format_to(dest, "blah {} blah {} blah", 123456, 0x123456));
    BM(, fmt::format_to(dest, "blah {:+} blah {:#x} blah", 123456, 0x123456));
    BM(, fmt::format_to(dest, "blah {:_>+20} blah {:<#20x} blah", 123456, 0x123456));

    BM(FIXTURE_STR, std::sprintf(dest, "Blah %s!\n", str.c_str()));
    BM(, std::sprintf(dest, "blah %d blah %d blah", 123456, 0x123456));
    BM(, std::sprintf(dest, "blah %+d blah %#x blah", 123456, 0x123456));
    BM(, std::sprintf(dest, "blah %+20d blah %#-20x blah", 123456, 0x123456));

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    strf::to(stdout)
        ( "\n where :"
          "\n    " STR(FIXTURE_STR)
          "\n    " STR(FMT_FIXTURE_STR)
          "\n    " STR(FMT_FIXTURE1)
          "\n    " STR(FMT_FIXTURE2)
          "\n    " STR(FMT_FIXTURE3)
          "\n" );

    return 0;
}
