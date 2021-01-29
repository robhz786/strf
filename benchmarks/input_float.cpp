#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS

#include <strf/to_cfile.hpp>
#define FMT_USE_GRISU 1
#include "fmt/compile.h"
#include "fmt/format.h"
#include <cstdio>
#include <cmath>
#include <clocale>
#include <benchmark/benchmark.h>

#define STR2(X) #X
#define STR(X) STR2(X)
#define CAT2(X, Y) X##Y
#define CAT(X,Y) CAT2(X,Y)
#define BM(FIXTURE, EXPR)  BM2(FIXTURE, EXPR, STR(EXPR))
#define BM2(FIXTURE, EXPR, LABEL)  DO_BM(CAT(bm_, __LINE__), FIXTURE, EXPR, LABEL)
#define DO_BM(ID, FIXTURE, EXPR, LABEL)                                   \
    struct ID {                                                         \
        static void func(benchmark::State& state) {                     \
            FIXTURE;                                                    \
            char dest[110];                                             \
            char* dest_end = dest + sizeof(dest);                       \
            (void) dest_end;                                            \
            for(auto _ : state) {                                       \
                EXPR ;                                                  \
                benchmark::DoNotOptimize(dest);                         \
                benchmark::ClobberMemory();                             \
            }                                                           \
        }                                                               \
    };                                                                  \
    benchmark::RegisterBenchmark(LABEL, ID :: func);

#define STRF_PUNCT_FIXTURE       auto punct_grp   = strf::numpunct<10>(3);
#define STRF_PUNCT_POINT_FIXTURE auto punct_point = strf::no_grouping<10>().decimal_point(':')

constexpr double pi = M_PI;


int main(int argc, char** argv)
{
    BM(, strf::to(dest)(1.11));
    BM(, strf::to(dest)(+strf::fmt(1.11)));
    BM(, strf::to(dest)(+strf::pad0(1.11, 20)));
    BM(, strf::to(dest)(+strf::right(1.11, 20)));
    BM(, strf::to(dest)(strf::sci(1.11).p(6)));
    BM(, strf::to(dest)(strf::fixed(1.11).p(6)));
    BM(, strf::to(dest)(strf::fmt(1.11).p(6)));

    BM(, strf::to(dest)(pi));
    BM(, strf::to(dest)(+strf::fmt(pi)));
    BM(, strf::to(dest)(+strf::pad0(pi, 20)));
    BM(, strf::to(dest)(+strf::right(pi, 20)));
    BM(, strf::to(dest)(strf::sci(pi).p(6)));
    BM(, strf::to(dest)(strf::fixed(pi).p(6)));
    BM(, strf::to(dest)(strf::fmt(pi).p(6)));


    BM(, strf::to(dest)(1.11e+50));
    BM(, strf::to(dest)(+strf::fmt(1.11e+50)));
    BM(, strf::to(dest)(+strf::pad0(1.11e+50, 20)));
    BM(, strf::to(dest)(+strf::right(1.11e+50, 20)));

    BM(, strf::to(dest)(strf::hex(1.11)));
    BM(, strf::to(dest)(strf::hex(pi)));
    BM(, strf::to(dest)(strf::hex(0.0)));
    BM(, strf::to(dest)(strf::hex(123456.0)));

    BM(, strf::to(dest)(strf::fixed(1000000.0)));

    BM(STRF_PUNCT_FIXTURE, strf::to(dest).with(punct_grp)(strf::fixed(1000000.0)));
    BM(STRF_PUNCT_POINT_FIXTURE, strf::to(dest).with(punct_point)(strf::fixed(1000000.0)));

    BM2(, fmt::format_to(dest, FMT_COMPILE("{}"), 1.11)
        , "fmt::format_to(dest, FMT_COMPILE(\"{}\"), 1.11)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:+}"), 1.11)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:+}\"), 1.11)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:+020}"), 1.11)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:+020}\"), 1.11)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:>+20}"), 1.11)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:>+20}\"), 1.11)");

    BM2(, fmt::format_to(dest, FMT_COMPILE("{:e}"), 1.11)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:e}\"), 1.11)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:f}"), 1.11)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:f}\"), 1.11)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:g}"), 1.11)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:g}\"), 1.11)");

    BM2(, fmt::format_to(dest, FMT_COMPILE("{}"), pi)
        , "fmt::format_to(dest, FMT_COMPILE(\"{}\"), pi)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:+}"), pi)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:+}\"), pi)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:+020}"), pi)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:+020}\"), pi)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:>+20}"), pi)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:>+20}\"), pi)");

    BM2(, fmt::format_to(dest, FMT_COMPILE("{:e}"), pi)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:e}\"), pi)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:f}"), pi)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:f}\"), pi)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:g}"), pi)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:g}\"), pi)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:a}"), pi)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:a}\"), pi)");

    BM2(, fmt::format_to(dest, FMT_COMPILE("{:}"), 1.11e+50)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:}\"), 1.11e+50)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:+}"), 1.11e+50)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:+}\"), 1.11e+50)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:+020}"), 1.11e+50)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:+020}\"), 1.11e+50)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:>+20}"), 1.11e+50)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:>+20}\"), 1.11e+50)");

    BM2(, fmt::format_to(dest, FMT_COMPILE("{:f}"), 1000000.0)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:f}\"), 1000000.0)");

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    strf::to(stdout)("----------------------------------------------------------------------- \n"
                     "    constexpr double pi = M_PI;\n"
                     "    " STR(STRF_PUNCT_FIXTURE) "\n"
                     "    " STR(STRF_PUNCT_POINT_FIXTURE) "\n"
                     "----------------------------------------------------------------------- \n");

    return 0;
}
