#define _CRT_SECURE_NO_WARNINGS

#include <strf/to_cfile.hpp>
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

int main(int argc, char** argv)
{
    BM(, strf::to(dest)(25));
    BM(, strf::to(dest)(15, 25, 35, 45, 55));
    BM(, strf::to(dest)(~0ull));
    BM(, strf::to(dest)(~0ull, ~0ull>>1, ~0ull>>2, ~0ull>>3, ~0ull>>4));
    BM(, strf::to(dest)(strf::dec(21).fill('*')<8, ' ', *strf::hex(221), ' ', +strf::dec(21)>10));

    BM2(,  fmt::format_to(dest, FMT_COMPILE( "{}" ), 25)
        , "fmt::format_to(dest, FMT_COMPILE(\"{}\"), 25)");
    BM2(,  fmt::format_to(dest, FMT_COMPILE( "{}{}{}{}{}" ), 15, 25, 35, 45, 55)
        , "fmt::format_to(dest, FMT_COMPILE(\"{}{}{}{}{}\"), 15, 25, 35, 45, 55);");
    BM2(,  fmt::format_to(dest, FMT_COMPILE( "{}" ), ~0ull)
        , "fmt::format_to(dest, FMT_COMPILE(\"{}\"), ~0ull)");
    BM2(,  fmt::format_to(dest, FMT_COMPILE( "{}{}{}{}{}" ), ~0ull, ~0ull>>1, ~0ull>>2, ~0ull>>3, ~0ull>>4)
        , "fmt::format_to(dest, FMT_COMPILE(\"{}{}{}{}{}\"), ~0ull, ~0ull>>1, ~0ull>>2, ~0ull>>3, ~0ull>>4)");
    BM2(,  fmt::format_to(dest, FMT_COMPILE( "{:*<8}{:#x}{:>+10}" ), 21, 221, 21)
        , "fmt::format_to(dest, FMT_COMPILE(\"{:*<8}{:#x}{:>+10}\"), 21, 221, 21)");

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
