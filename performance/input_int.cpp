#define _CRT_SECURE_NO_WARNINGS

#include <strf.hpp>
#include "fmt/format.h"
#include "fmt/compile.h"
#include <benchmark/benchmark.h>
#include <cstdio>
#include <climits>
#include <clocale>

#if defined(__has_include)
#if __has_include(<charconv>)
#if (__cplusplus >= 201402L) || (defined(_HAS_CXX17) && _HAS_CXX17)
#define HAS_CHARCONV
#include <charconv>
#endif
#endif
#endif


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

#define STRF_I_OP         strf::to(dest)(25);
#define FMT_I_FIXTURE     auto fmt_int = fmt::compile<int>("{}");
#define FMT_I_OP          fmt::format_to(dest, fmt_int, 25);
#define FMT_INT_I_OP      strcpy(dest, fmt::format_int{25}.c_str());
#define TOCHARS_I_OP      std::to_chars(dest, dest_end, 25);
#define SPRINTF_I_OP      std::sprintf(dest, "%d", 25);
#define STRF_5I_OP        strf::to(dest)(15, 25, 35, 45, 55);
#define FMT_5I_FIXTURE    auto fmt_5x_int = fmt::compile<int, int, int, int, int>("{}{}{}{}{}");
#define FMT_5I_OP         fmt::format_to(dest + 10, fmt_5x_int, 15, 25, 35, 45, 55);
#define SPRINTF_5I_OP     std::sprintf(dest, "%d%d%d%d%d", 15, 25, 35, 45, 55);
#define STRF_ULL_OP       strf::to(dest)(~0ull);
#define FMT_ULL_FIXTURE   auto fmt_ulonglong = fmt::compile<std::uint64_t>("{}");
#define FMT_ULL_OP        fmt::format_to(dest, fmt_ulonglong, ~0ull);
#define FMT_INT_ULL_OP    strcpy(dest, fmt::format_int{~0ull}.c_str());
#define TOCHARS_ULL_OP    std::to_chars(dest, dest_end, ~0ull);
#define SPRINTF_ULL_OP    std::sprintf(dest, "%lld", LLONG_MAX);
#define STRF_5ULL_OP      strf::to(dest)(~0ull, ~0ull>>1, ~0ull>>2, ~0ull>>3, ~0ull>>4);
#define FMT_5ULL_FIXTURE  auto fmt_5x_ulonglong = \
                          fmt::compile<std::uint64_t, std::uint64_t, std::uint64_t, \
                                       std::uint64_t, std::uint64_t>("{}{}{}{}{}");
#define FMT_5ULL_OP       fmt::format_to(dest, fmt_5x_ulonglong, ~0ull, ~0ull>>1, ~0ull>>2, ~0ull>>3, ~0ull>>4);
#define SPRINTF_5ULL_OP   std::sprintf(dest, "%llud%llud%llud%llud%llud", ~0ull, ~0ull>>1, ~0ull>>2, ~0ull>>3, ~0ull>>4 );
#define STRF_MIX_OP       strf::to(dest)(strf::dec(21).fill('*')<8, ' ', ~strf::hex(221), ' ', +strf::dec(21)>10);
#define FMT_MIX_FIXTURE   auto fmt_3_int = fmt::compile<int, int, int>("{:*<8}{:#x}{:>+10}");
#define FMT_MIX_OP        fmt::format_to(dest, fmt_3_int, 21, 221, 21);
#define SPRINTF_MIX_OP    std::sprintf(dest, "%-8d%#x%+10d", 21, 221, 21);
#define STRF_PUNCT_FIXTURE    strf::monotonic_grouping<10> punct3(3);
#define STRF_PUNCT_I_OP       strf::to(dest).with(punct3)(25);
#define FMT_PUNCT_I_FIXTURE   std::setlocale(LC_ALL, "en_US.UTF-8"); \
                              auto fmt_loc_int = fmt::compile<int>("{:n}");
#define FMT_PUNCT_I_OP        fmt::format_to(dest, fmt_loc_int, 25);
#define SPRINTF_PUNCT_I_OP    std::sprintf(dest, "%'d", 25);
#define STRF_PUNCT_ULL_OP     strf::to(dest).with(punct3)(~0ull);
#define FMT_PUNCT_ULL_FIXTURE std::setlocale(LC_ALL, "en_US.UTF-8"); \
                              auto fmt_loc_ulonglong = fmt::compile<std::uint64_t>("{:n}");
#define FMT_PUNCT_ULL_OP      fmt::format_to(dest, fmt_loc_ulonglong, ~0ull);
#define SPRINTF_PUNCT_ULL_OP  std::sprintf(dest, "%'llud", ~0ull);
#define STRF_BIGPUNCT_FIXTURE auto punct3_bigsep = strf::monotonic_grouping<10>(3).thousands_sep(0x22C4);
#define STRF_BIGPUNCT_I_OP    strf::to(dest).with(punct3_bigsep)(25);
#define STRF_BIGPUNCT_ULL_OP  strf::to(dest).with(punct3_bigsep)(~0ull);


CREATE_BENCHMARK(STRF_I, ;);
CREATE_BENCHMARK(FMT_I, FMT_I_FIXTURE);
CREATE_BENCHMARK(FMT_INT_I,  ;);
#if defined(HAS_CHARCONV)
CREATE_BENCHMARK(TOCHARS_I,  ;);
#endif
CREATE_BENCHMARK(SPRINTF_I,  ;);
CREATE_BENCHMARK(STRF_5I,  ;);
CREATE_BENCHMARK(FMT_5I, FMT_5I_FIXTURE);
CREATE_BENCHMARK(SPRINTF_5I,  ;);
CREATE_BENCHMARK(STRF_ULL,  ;);
CREATE_BENCHMARK(FMT_ULL, FMT_ULL_FIXTURE);
CREATE_BENCHMARK(FMT_INT_ULL,  ;);
#if defined(HAS_CHARCONV)
CREATE_BENCHMARK(TOCHARS_ULL,  ;);
#endif
CREATE_BENCHMARK(SPRINTF_ULL,  ;);
CREATE_BENCHMARK(STRF_5ULL,  ;);
CREATE_BENCHMARK(FMT_5ULL,  FMT_5ULL_FIXTURE);
CREATE_BENCHMARK(SPRINTF_5ULL,  ;);
CREATE_BENCHMARK(STRF_MIX,  ;);
CREATE_BENCHMARK(FMT_MIX, FMT_MIX_FIXTURE);
CREATE_BENCHMARK(SPRINTF_MIX,  ;);
CREATE_BENCHMARK(STRF_PUNCT_I, STRF_PUNCT_FIXTURE);
CREATE_BENCHMARK(FMT_PUNCT_I, FMT_PUNCT_I_FIXTURE);
#if defined(__GNU_LIBRARY__)
CREATE_BENCHMARK(SPRINTF_PUNCT_I, ;);
#endif
CREATE_BENCHMARK(STRF_PUNCT_ULL, STRF_PUNCT_FIXTURE);
CREATE_BENCHMARK(FMT_PUNCT_ULL,  FMT_PUNCT_ULL_FIXTURE);
#if defined(__GNU_LIBRARY__)
CREATE_BENCHMARK(SPRINTF_PUNCT_ULL, ;);
#endif
CREATE_BENCHMARK(STRF_BIGPUNCT_I, STRF_BIGPUNCT_FIXTURE);
CREATE_BENCHMARK(STRF_BIGPUNCT_ULL, STRF_BIGPUNCT_FIXTURE);


int main(int argc, char** argv)
{
    REGISTER_BENCHMARK(STRF_I);
    REGISTER_BENCHMARK(FMT_I);
    REGISTER_BENCHMARK(FMT_INT_I);
#if defined(HAS_CHARCONV)
    REGISTER_BENCHMARK(TOCHARS_I);
#endif
    REGISTER_BENCHMARK(SPRINTF_I);
    REGISTER_BENCHMARK(STRF_5I);
    REGISTER_BENCHMARK(FMT_5I);
    REGISTER_BENCHMARK(SPRINTF_5I);
    REGISTER_BENCHMARK(STRF_ULL);
    REGISTER_BENCHMARK(FMT_ULL);
    REGISTER_BENCHMARK(FMT_INT_ULL);
#if defined(HAS_CHARCONV)
    REGISTER_BENCHMARK(TOCHARS_ULL);
#endif
    REGISTER_BENCHMARK(SPRINTF_ULL);
    REGISTER_BENCHMARK(STRF_5ULL);
    REGISTER_BENCHMARK(FMT_5ULL);
    REGISTER_BENCHMARK(SPRINTF_5ULL);
    REGISTER_BENCHMARK(STRF_MIX);
    REGISTER_BENCHMARK(FMT_MIX);
    REGISTER_BENCHMARK(SPRINTF_MIX);
    REGISTER_BENCHMARK(STRF_PUNCT_I);
    REGISTER_BENCHMARK(FMT_PUNCT_I);
#if defined(__GNU_LIBRARY__)
    REGISTER_BENCHMARK(SPRINTF_PUNCT_I);
#endif
    REGISTER_BENCHMARK(STRF_PUNCT_ULL);
    REGISTER_BENCHMARK(FMT_PUNCT_ULL);
#if defined(__GNU_LIBRARY__)
    REGISTER_BENCHMARK(SPRINTF_PUNCT_ULL);
#endif
    REGISTER_BENCHMARK(STRF_BIGPUNCT_I);
    REGISTER_BENCHMARK(STRF_BIGPUNCT_ULL);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

   strf::to(stdout)(
        "auto fmt_int = fmt::compile<int>(\"{}\");\n"
        "auto fmt_5x_int = fmt::compile<int, int, int, int, int>(\"{}{}{}{}{}\");\n"
        "auto fmt_ulonglong = fmt::compile<std::uint64_t>(\"{}\");\n"
        "auto fmt_5x_ulonglong = fmt::compile<std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t>(\"{}{}{}{}{}\");\n"
        "auto fmt_3_int = fmt::compile<int, int, int>(\"{:*<8}{:#x}{:>+10}\");\n"
        "auto fmt_loc_int = fmt::compile<int>(\"{:n}\");\n"
        "auto fmt_loc_ulonglong = fmt::compile<std::uint64_t>(\"{:n}\");\n" );

    return 0;
}
