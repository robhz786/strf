#include <strf/to_cfile.hpp>
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
            char dest[60];                                              \
            char* dest_end = dest + sizeof(dest);                       \
            (void) dest_end;                                            \
            for(auto _ : state) {                                       \
                EXPR ;                                                  \
                benchmark::DoNotOptimize(dest);                         \
                benchmark::ClobberMemory();                             \
            }                                                           \
        }                                                               \
    };                                                                  \
    benchmark::RegisterBenchmark(STR(EXPR), ID :: func);


#define MP1 const auto mp1 = strf::numpunct<10>(3);
#define MP2 const auto mp2 = strf::numpunct<10>(3).thousands_sep(0xb7);
#define SP1 const auto sp1 = strf::numpunct<10>(2, 4, 3);
#define SP2 const auto sp2 = strf::numpunct<10>(2, 4, 3).thousands_sep(0xb7);

int main(int argc, char** argv)
{
    BM(MP1, strf::to(dest).with(mp1) (1000ull));
    BM(MP1, strf::to(dest).with(mp1) (1000000000ull));
    BM(MP1, strf::to(dest).with(mp1) (1000000000000000000ull));
    BM(MP1, strf::to(dest).with(mp1) (strf::fixed(1e+3)));
    BM(MP1, strf::to(dest).with(mp1) (strf::fixed(1e+12)));
    BM(MP1, strf::to(dest).with(mp1) (strf::fixed(1e+120)));

    BM(MP2, strf::to(dest).with(mp2) (1000ull));
    BM(MP2, strf::to(dest).with(mp2) (1000000000ull));
    BM(MP2, strf::to(dest).with(mp2) (1000000000000000000ull));
    BM(MP2, strf::to(dest).with(mp2) (strf::fixed(1e+3)));
    BM(MP2, strf::to(dest).with(mp2) (strf::fixed(1e+18)));
    BM(MP2, strf::to(dest).with(mp2) (strf::fixed(1e+180)));

    BM(SP1, strf::to(dest).with(sp1) (1000ull));
    BM(SP1, strf::to(dest).with(sp1) (1000000000ull));
    BM(SP1, strf::to(dest).with(sp1) (1000000000000000000ull));
    BM(SP1, strf::to(dest).with(sp1) (strf::fixed(1e+3)));
    BM(SP1, strf::to(dest).with(sp1) (strf::fixed(1e+12)));
    BM(SP1, strf::to(dest).with(sp1) (strf::fixed(1e+120)));

    BM(SP2, strf::to(dest).with(sp2) (1000ull));
    BM(SP2, strf::to(dest).with(sp2) (1000000000ull));
    BM(SP2, strf::to(dest).with(sp2) (1000000000000000000ull));
    BM(SP2, strf::to(dest).with(sp2) (strf::fixed(1e+3)));
    BM(SP2, strf::to(dest).with(sp2) (strf::fixed(1e+18)));
    BM(SP2, strf::to(dest).with(sp2) (strf::fixed(1e+180)));

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    strf::to(stdout)(
        "\n    " STR(MP1)
        "\n    " STR(MP2)
        "\n    " STR(SP1)
        "\n    " STR(SP2) "\n" );

    return 0;
}
