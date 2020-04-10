#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS

#include <strf.hpp>
#define FMT_USE_GRISU 1
#include "fmt/compile.h"
#include "fmt/format.h"
#include <cstdio>
#include <cmath>
#include <clocale>
#include <benchmark/benchmark.h>

#define CREATE_BENCHMARK(PREFIX, FIXTURE)                    \
    static void PREFIX ## _func (benchmark::State& state) {  \
        FIXTURE;                                             \
        char dest[110];                                      \
        char* dest_end = dest + sizeof(dest);                \
        (void) dest_end;                                     \
        for(auto _ : state) {                                \
            PREFIX ## _OP ;                                  \
            benchmark::DoNotOptimize(dest);                  \
            benchmark::ClobberMemory();                      \
        }                                                    \
    }

#define REGISTER_BENCHMARK(X) benchmark::RegisterBenchmark(STR(X ## _OP), X ## _func);
#define STR2(X) #X
#define STR(X) STR2(X)

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

#define FMT_D1_OP        fmt::format_to(dest, fmtd, 1.11)
#define FMT_E1_OP        fmt::format_to(dest, fmte, 1.11)
#define FMT_F1_OP        fmt::format_to(dest, fmtf, 1.11)
#define FMT_G1_OP        fmt::format_to(dest, fmtg, 1.11)
#define FMT_D_PI_OP      fmt::format_to(dest, fmtd, pi)
#define FMT_E_PI_OP      fmt::format_to(dest, fmte, pi)
#define FMT_F_PI_OP      fmt::format_to(dest, fmtf, pi)
#define FMT_G_PI_OP      fmt::format_to(dest, fmtg, pi)
#define FMT_X_PI_OP      fmt::format_to(dest, fmta, pi)
#define FMT_F_BIG_OP     fmt::format_to(dest, fmtf, 1000000.0)
//#define FMT_PUNCT_OP     fmt::format_to(dest, fmt_loc, 1000000.0)

#define SPRINTF_E1_OP    std::sprintf(dest, "%e", 1.11)
#define SPRINTF_F1_OP    std::sprintf(dest, "%f", 1.11)
#define SPRINTF_G1_OP    std::sprintf(dest, "%g", 1.11)
#define SPRINTF_E_PI_OP  std::sprintf(dest, "%e", pi)
#define SPRINTF_F_PI_OP  std::sprintf(dest, "%f", pi)
#define SPRINTF_G_PI_OP  std::sprintf(dest, "%g", pi)
#define SPRINTF_X_PI_OP  std::sprintf(dest, "%a", pi)
#define SPRINTF_F_BIG_OP std::sprintf(dest, "%f", 1000000.0)

#define STRF_PUNCT_FIXTURE       auto punct_grp   = strf::monotonic_grouping<10>(3);
#define STRF_PUNCT_POINT_FIXTURE auto punct_point = strf::no_grouping<10>().decimal_point(':')

#define FMTD_FIXTURE    auto fmtd    = fmt::compile<double>("{}")
#define FMTE_FIXTURE    auto fmte    = fmt::compile<double>("{:e}")
#define FMTF_FIXTURE    auto fmtf    = fmt::compile<double>("{:f}")
#define FMTG_FIXTURE    auto fmtg    = fmt::compile<double>("{:g}")
#define FMTA_FIXTURE    auto fmta    = fmt::compile<double>("{:a}")
//#define FMT_LOC_FIXTURE auto fmt_loc = fmt::compile<double>("{:fn}")

constexpr double pi = M_PI;

CREATE_BENCHMARK(STRF_D1, ;);
CREATE_BENCHMARK(STRF_E1, ;);
CREATE_BENCHMARK(STRF_F1, ;);
CREATE_BENCHMARK(STRF_G1, ;);

CREATE_BENCHMARK(STRF_D_PI, ;);
CREATE_BENCHMARK(STRF_E_PI, ;);
CREATE_BENCHMARK(STRF_F_PI, ;);
CREATE_BENCHMARK(STRF_G_PI, ;);
CREATE_BENCHMARK(STRF_X_PI, ;);
CREATE_BENCHMARK(STRF_F_BIG, ;);
CREATE_BENCHMARK(STRF_PUNCT, STRF_PUNCT_FIXTURE);
CREATE_BENCHMARK(STRF_POINT, STRF_PUNCT_POINT_FIXTURE);

CREATE_BENCHMARK(FMT_D1, FMTD_FIXTURE);
CREATE_BENCHMARK(FMT_E1, FMTE_FIXTURE);
CREATE_BENCHMARK(FMT_F1, FMTF_FIXTURE);
CREATE_BENCHMARK(FMT_G1, FMTG_FIXTURE);
CREATE_BENCHMARK(FMT_D_PI, FMTD_FIXTURE);
CREATE_BENCHMARK(FMT_E_PI, FMTE_FIXTURE);
CREATE_BENCHMARK(FMT_F_PI, FMTF_FIXTURE);
CREATE_BENCHMARK(FMT_G_PI, FMTG_FIXTURE);
CREATE_BENCHMARK(FMT_X_PI, FMTA_FIXTURE);
CREATE_BENCHMARK(FMT_F_BIG, FMTF_FIXTURE);
//CREATE_BENCHMARK(FMT_PUNCT, FMT_LOC_FIXTURE);

CREATE_BENCHMARK(SPRINTF_E1, ;);
CREATE_BENCHMARK(SPRINTF_F1, ;);
CREATE_BENCHMARK(SPRINTF_G1, ;);

CREATE_BENCHMARK(SPRINTF_E_PI, ;);
CREATE_BENCHMARK(SPRINTF_F_PI, ;);
CREATE_BENCHMARK(SPRINTF_G_PI, ;);
CREATE_BENCHMARK(SPRINTF_X_PI, ;);
CREATE_BENCHMARK(SPRINTF_F_BIG, ;);

int main(int argc, char** argv)
{
    REGISTER_BENCHMARK(STRF_D1);
    REGISTER_BENCHMARK(STRF_E1);
    REGISTER_BENCHMARK(STRF_F1);
    REGISTER_BENCHMARK(STRF_G1);

    REGISTER_BENCHMARK(STRF_D_PI);
    REGISTER_BENCHMARK(STRF_E_PI);
    REGISTER_BENCHMARK(STRF_F_PI);
    REGISTER_BENCHMARK(STRF_G_PI);
    REGISTER_BENCHMARK(STRF_X_PI);

    REGISTER_BENCHMARK(STRF_F_BIG);
    REGISTER_BENCHMARK(STRF_PUNCT);
    REGISTER_BENCHMARK(STRF_POINT);

    REGISTER_BENCHMARK(FMT_D1);
    REGISTER_BENCHMARK(FMT_E1);
    REGISTER_BENCHMARK(FMT_F1);
    REGISTER_BENCHMARK(FMT_G1);
    REGISTER_BENCHMARK(FMT_D_PI);
    REGISTER_BENCHMARK(FMT_E_PI);
    REGISTER_BENCHMARK(FMT_F_PI);
    REGISTER_BENCHMARK(FMT_G_PI);
    REGISTER_BENCHMARK(FMT_X_PI);
    REGISTER_BENCHMARK(FMT_F_BIG);
    // REGISTER_BENCHMARK(FMT_PUNCT);

    REGISTER_BENCHMARK(SPRINTF_E1);
    REGISTER_BENCHMARK(SPRINTF_F1);
    REGISTER_BENCHMARK(SPRINTF_G1);

    REGISTER_BENCHMARK(SPRINTF_E_PI);
    REGISTER_BENCHMARK(SPRINTF_F_PI);
    REGISTER_BENCHMARK(SPRINTF_G_PI);
    REGISTER_BENCHMARK(SPRINTF_X_PI);
    REGISTER_BENCHMARK(SPRINTF_F_BIG);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    strf::to(stdout)("----------------------------------------------------------------------- \n"
                     "    constexpr double pi = M_PI;\n"
                     "    " STR(STRF_PUNCT_FIXTURE) "\n"
                     "    " STR(STRF_PUNCT_POINT_FIXTURE) "\n"
                     "    " STR(FMTD_FIXTURE) "\n"
                     "    " STR(FMTE_FIXTURE) "\n"
                     "    " STR(FMTF_FIXTURE) "\n"
                     "    " STR(FMTG_FIXTURE) "\n"
                     "    " STR(FMTA_FIXTURE) "\n"
                     "----------------------------------------------------------------------- \n");

    return 0;
}
