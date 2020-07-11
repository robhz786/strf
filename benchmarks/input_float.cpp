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

#define STRF_D1_OP       strf::to(dest)(1.11)
#define STRF_E1_OP       strf::to(dest)(strf::sci(1.11).p(6))
#define STRF_F1_OP       strf::to(dest)(strf::fixed(1.11).p(6))
#define STRF_G1_OP       strf::to(dest)(strf::fmt(1.11).p(6))
#define STRF_D_PI_OP     strf::to(dest)(pi)
#define STRF_E_PI_OP     strf::to(dest)(strf::sci(pi).p(6))
#define STRF_F_PI_OP     strf::to(dest)(strf::fixed(pi).p(6))
#define STRF_G_PI_OP     strf::to(dest)(strf::fmt(pi).p(6))
#define STRF_X_PI_OP     strf::to(dest)(strf::hex(pi).p(6))
#define STRF_F_BIG_OP    strf::to(dest)(strf::fixed(1000000.0))
#define STRF_PUNCT_OP    strf::to(dest).with(punct_grp)(strf::fixed(1000000.0))
#define STRF_POINT_OP    strf::to(dest).with(punct_point)(strf::fixed(1000000.0))

#define FMT_D1_OP        fmt::format_to(dest, FMT_COMPILE("{}"), 1.11)
#define FMT_E1_OP        fmt::format_to(dest, FMT_COMPILE("{:e}"), 1.11)
#define FMT_F1_OP        fmt::format_to(dest, FMT_COMPILE("{:f}"), 1.11)
#define FMT_G1_OP        fmt::format_to(dest, FMT_COMPILE("{:g}"), 1.11)
#define FMT_D_PI_OP      fmt::format_to(dest, FMT_COMPILE("{}"), pi)
#define FMT_E_PI_OP      fmt::format_to(dest, FMT_COMPILE("{:e}"), pi)
#define FMT_F_PI_OP      fmt::format_to(dest, FMT_COMPILE("{:f}"), pi)
#define FMT_G_PI_OP      fmt::format_to(dest, FMT_COMPILE("{:g}"), pi)
#define FMT_X_PI_OP      fmt::format_to(dest, FMT_COMPILE("{:a}"), pi)
#define FMT_F_BIG_OP     fmt::format_to(dest, FMT_COMPILE("{:f}"), 1000000.0)
//#define FMT_PUNCT_OP     fmt::format_to(dest, FMT_COMPILE("{:fn}"), 1000000.0)

#define SPRINTF_E1_OP    std::sprintf(dest, "%e", 1.11)
#define SPRINTF_F1_OP    std::sprintf(dest, "%f", 1.11)
#define SPRINTF_G1_OP    std::sprintf(dest, "%g", 1.11)
#define SPRINTF_E_PI_OP  std::sprintf(dest, "%e", pi)
#define SPRINTF_F_PI_OP  std::sprintf(dest, "%f", pi)
#define SPRINTF_G_PI_OP  std::sprintf(dest, "%g", pi)
#define SPRINTF_X_PI_OP  std::sprintf(dest, "%a", pi)
#define SPRINTF_F_BIG_OP std::sprintf(dest, "%f", 1000000.0)

#define STRF_PUNCT_FIXTURE       auto punct_grp   = strf::numpunct<10>(3);
#define STRF_PUNCT_POINT_FIXTURE auto punct_point = strf::no_grouping<10>().decimal_point(':')

constexpr double pi = M_PI;


int main(int argc, char** argv)
{
    BM(, strf::to(dest)(1.11));
    BM(, strf::to(dest)(strf::sci(1.11).p(6)));
    BM(, strf::to(dest)(strf::fixed(1.11).p(6)));
    BM(, strf::to(dest)(strf::fmt(1.11).p(6)));
    BM(, strf::to(dest)(pi));
    BM(, strf::to(dest)(strf::sci(pi).p(6)));
    BM(, strf::to(dest)(strf::fixed(pi).p(6)));
    BM(, strf::to(dest)(strf::fmt(pi).p(6)));
    BM(, strf::to(dest)(strf::hex(pi).p(6)));
    BM(, strf::to(dest)(strf::fixed(1000000.0)));
    BM(STRF_PUNCT_FIXTURE, strf::to(dest).with(punct_grp)(strf::fixed(1000000.0)));
    BM(STRF_PUNCT_POINT_FIXTURE, strf::to(dest).with(punct_point)(strf::fixed(1000000.0)));

    BM2(, fmt::format_to(dest, FMT_COMPILE("{}"), 1.11)
        , "fmt::format_to(dest, FMT_COMPILE(\"{}\"), 1.11)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:e}"), 1.11)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:e}\"), 1.11)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:f}"), 1.11)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:f}\"), 1.11)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:g}"), 1.11)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:g}\"), 1.11)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{}"), pi)
        , "fmt::format_to(dest, FMT_COMPILE(\"{}\"), pi)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:e}"), pi)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:e}\"), pi)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:f}"), pi)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:f}\"), pi)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:g}"), pi)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:g}\"), pi)");
    BM2(, fmt::format_to(dest, FMT_COMPILE("{:a}"), pi)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:a}\"), pi)");
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
