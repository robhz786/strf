#include <strf.hpp>
#include <benchmark/benchmark.h>

#define CREATE_BENCHMARK(PREFIX, FIXTURE)                               \
    static void PREFIX ## _func (benchmark::State& state) {             \
        FIXTURE;                                                        \
        char dest[60];                                                  \
        char* dest_end = dest + sizeof(dest);                           \
        (void) dest_end;                                                \
        for(auto _ : state) {                                           \
            PREFIX ## _OP ;                                             \
            benchmark::DoNotOptimize(dest);                             \
            benchmark::ClobberMemory();                                 \
        }                                                               \
    }

#define REGISTER_BENCHMARK(X) benchmark::RegisterBenchmark(STR(X ## _OP), X ## _func);
#define STR2(X) #X
#define STR(X) STR2(X)

#define MP1_FIXTURE const auto mp1 = strf::monotonic_grouping<10>{3};
#define MP2_FIXTURE const auto mp2 = strf::monotonic_grouping<10>{3}.thousands_sep(0xb7);
#define SP1_FIXTURE const auto sp1 = strf::str_grouping<10>{"\002\004\003"};
#define SP2_FIXTURE const auto sp2 = strf::str_grouping<10>{"\002\004\003"}.thousands_sep(0xb7);

#define MP1_I1_OP  strf::to(dest).with(mp1) (1000ull);
#define MP1_I3_OP  strf::to(dest).with(mp1) (1000000000ull);
#define MP1_I6_OP  strf::to(dest).with(mp1) (1000000000000000000ull);
#define MP1_F1_OP  strf::to(dest).with(mp1) (strf::fixed(1e+3));
#define MP1_F6_OP  strf::to(dest).with(mp1) (strf::fixed(1e+12));
#define MP1_F60_OP strf::to(dest).with(mp1) (strf::fixed(1e+120));

#define MP2_I1_OP  strf::to(dest).with(mp2) (1000ull);
#define MP2_I3_OP  strf::to(dest).with(mp2) (1000000000ull);
#define MP2_I6_OP  strf::to(dest).with(mp2) (1000000000000000000ull);
#define MP2_F1_OP  strf::to(dest).with(mp2) (strf::fixed(1e+3));
#define MP2_F6_OP  strf::to(dest).with(mp2) (strf::fixed(1e+18));
#define MP2_F60_OP strf::to(dest).with(mp2) (strf::fixed(1e+180));

#define SP1_I1_OP  strf::to(dest).with(sp1) (1000ull);
#define SP1_I3_OP  strf::to(dest).with(sp1) (1000000000ull);
#define SP1_I6_OP  strf::to(dest).with(sp1) (1000000000000000000ull);
#define SP1_F1_OP  strf::to(dest).with(sp1) (strf::fixed(1e+3));
#define SP1_F6_OP  strf::to(dest).with(sp1) (strf::fixed(1e+12));
#define SP1_F60_OP strf::to(dest).with(sp1) (strf::fixed(1e+120));

#define SP2_I1_OP  strf::to(dest).with(sp2) (1000ull);
#define SP2_I3_OP  strf::to(dest).with(sp2) (1000000000ull);
#define SP2_I6_OP  strf::to(dest).with(sp2) (1000000000000000000ull);
#define SP2_F1_OP  strf::to(dest).with(sp2) (strf::fixed(1e+3));
#define SP2_F6_OP  strf::to(dest).with(sp2) (strf::fixed(1e+18));
#define SP2_F60_OP strf::to(dest).with(sp2) (strf::fixed(1e+180));

CREATE_BENCHMARK(MP1_I1, MP1_FIXTURE);
CREATE_BENCHMARK(MP1_I3, MP1_FIXTURE);
CREATE_BENCHMARK(MP1_I6, MP1_FIXTURE);
CREATE_BENCHMARK(MP1_F1, MP1_FIXTURE);
CREATE_BENCHMARK(MP1_F6, MP1_FIXTURE);
CREATE_BENCHMARK(MP1_F60, MP1_FIXTURE);

CREATE_BENCHMARK(MP2_I1, MP2_FIXTURE);
CREATE_BENCHMARK(MP2_I3, MP2_FIXTURE);
CREATE_BENCHMARK(MP2_I6, MP2_FIXTURE);
CREATE_BENCHMARK(MP2_F1, MP2_FIXTURE);
CREATE_BENCHMARK(MP2_F6, MP2_FIXTURE);
CREATE_BENCHMARK(MP2_F60, MP2_FIXTURE);

CREATE_BENCHMARK(SP1_I1, SP1_FIXTURE);
CREATE_BENCHMARK(SP1_I3, SP1_FIXTURE);
CREATE_BENCHMARK(SP1_I6, SP1_FIXTURE);
CREATE_BENCHMARK(SP1_F1, SP1_FIXTURE);
CREATE_BENCHMARK(SP1_F6, SP1_FIXTURE);
CREATE_BENCHMARK(SP1_F60, SP1_FIXTURE);

CREATE_BENCHMARK(SP2_I1, SP2_FIXTURE);
CREATE_BENCHMARK(SP2_I3, SP2_FIXTURE);
CREATE_BENCHMARK(SP2_I6, SP2_FIXTURE);
CREATE_BENCHMARK(SP2_F1, SP2_FIXTURE);
CREATE_BENCHMARK(SP2_F6, SP2_FIXTURE);
CREATE_BENCHMARK(SP2_F60, SP2_FIXTURE);


int main(int argc, char** argv)
{
    REGISTER_BENCHMARK(MP1_I1);
    REGISTER_BENCHMARK(MP1_I3);
    REGISTER_BENCHMARK(MP1_I6);
    REGISTER_BENCHMARK(MP1_F1);
    REGISTER_BENCHMARK(MP1_F6);
    REGISTER_BENCHMARK(MP1_F60);

    REGISTER_BENCHMARK(MP2_I1);
    REGISTER_BENCHMARK(MP2_I3);
    REGISTER_BENCHMARK(MP2_I6);
    REGISTER_BENCHMARK(MP2_F1);
    REGISTER_BENCHMARK(MP2_F6);
    REGISTER_BENCHMARK(MP2_F60);

    REGISTER_BENCHMARK(SP1_I1);
    REGISTER_BENCHMARK(SP1_I3);
    REGISTER_BENCHMARK(SP1_I6);
    REGISTER_BENCHMARK(SP1_F1);
    REGISTER_BENCHMARK(SP1_F6);
    REGISTER_BENCHMARK(SP1_F60);

    REGISTER_BENCHMARK(SP2_I1);
    REGISTER_BENCHMARK(SP2_I3);
    REGISTER_BENCHMARK(SP2_I6);
    REGISTER_BENCHMARK(SP2_F1);
    REGISTER_BENCHMARK(SP2_F6);
    REGISTER_BENCHMARK(SP2_F60);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    strf::to(stdout)(
        "\n    " STR(MP1_FIXTURE)
        "\n    " STR(MP2_FIXTURE)
        "\n    " STR(SP1_FIXTURE)
        "\n    " STR(SP2_FIXTURE) "\n" );

    return 0;
}
