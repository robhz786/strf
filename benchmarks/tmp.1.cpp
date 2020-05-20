#include <strf.hpp>
#include <benchmark/benchmark.h>

// template <std::size_t>
// static const char* bm_name();

// template <std::size_t>
// static void bm_func(benchmark::State& state);

#define STR2(X) #X
#define STR(X) STR2(X)

#define CREATE_BENCHMARK(ID, FIXTURE, EXPR)                             \
    inline const char* bm_name_ ## ID () { return STR(EXPR); }          \
    static void bm_func_ ## ID(benchmark::State& state) {               \
        FIXTURE;                                                        \
        char dest[60];                                                  \
        char* dest_end = dest + sizeof(dest);                           \
        (void) dest_end;                                                \
        for(auto _ : state) {                                           \
            EXPR ;                                                      \
            benchmark::DoNotOptimize(dest);                             \
            benchmark::ClobberMemory();                                 \
        }                                                               \
    }

#define REGISTER_BENCHMARK(ID) \
    benchmark::RegisterBenchmark(bm_name_ ## ID (), bm_func_ ## ID);

#define MP1_FIXTURE const auto mp1 = strf::numpunct<10>{3};
#define MP2_FIXTURE const auto mp2 = strf::numpunct<10>{3}.thousands_sep(0xb7);
#define SP1_FIXTURE const auto sp1 = strf::str_grouping<10>{"\002\004\003"};
#define SP2_FIXTURE const auto sp2 = strf::str_grouping<10>{"\002\004\003"}.thousands_sep(0xb7);

CREATE_BENCHMARK(mp1_i1, MP1_FIXTURE, strf::to(dest).with(mp1) (1000ull));


int main(int argc, char** argv)
{
    REGISTER_BENCHMARK(mp1_i1);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    strf::to(stdout)(
        "\n    " STR(MP1_FIXTURE)
        "\n    " STR(MP2_FIXTURE)
        "\n    " STR(SP1_FIXTURE)
        "\n    " STR(SP2_FIXTURE) "\n" );

    return 0;
}
