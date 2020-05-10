#include <strf.hpp>
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


#define MP1_DEC const auto mp1_dec = strf::numpunct<10>{3};
#define MP2_DEC const auto mp2_dec = strf::numpunct<10>{3}.thousands_sep(0xb7);
#define SP1_DEC const auto sp1_dec = strf::numpunct<10>{3};
#define SP2_DEC const auto sp2_dec = strf::numpunct<10>{3}.thousands_sep(0xb7);

#define MP1_HEX const auto mp1_hex = strf::numpunct<16>{3};
#define MP2_HEX const auto mp2_hex = strf::numpunct<16>{3}.thousands_sep(0xb7);
#define SP1_HEX const auto sp1_hex = strf::numpunct<16>{3};
#define SP2_HEX const auto sp2_hex = strf::numpunct<16>{3}.thousands_sep(0xb7);

#define MP1_OCT const auto mp1_oct = strf::numpunct<8>{3};
#define MP2_OCT const auto mp2_oct = strf::numpunct<8>{3}.thousands_sep(0xb7);
#define SP1_OCT const auto sp1_oct = strf::numpunct<8>{3};
#define SP2_OCT const auto sp2_oct = strf::numpunct<8>{3}.thousands_sep(0xb7);

#define MP1_BIN const auto mp1_bin = strf::numpunct<2>{3};
#define MP2_BIN const auto mp2_bin = strf::numpunct<2>{3}.thousands_sep(0xb7);
#define SP1_BIN const auto sp1_bin = strf::numpunct<2>{3};
#define SP2_BIN const auto sp2_bin = strf::numpunct<2>{3}.thousands_sep(0xb7);

int main(int argc, char** argv)
{
    BM(MP1_DEC, strf::to(dest).with(mp1_dec) (1000ull));
    BM(MP1_DEC, strf::to(dest).with(mp1_dec) (1000000000ull));
    BM(MP1_DEC, strf::to(dest).with(mp1_dec) (1000000000000000000ull));

    BM(MP2_DEC, strf::to(dest).with(mp2_dec) (1000ull));
    BM(MP2_DEC, strf::to(dest).with(mp2_dec) (1000000000ull));
    BM(MP2_DEC, strf::to(dest).with(mp2_dec) (1000000000000000000ull));

    BM(SP1_DEC, strf::to(dest).with(sp1_dec) (1000ull));
    BM(SP1_DEC, strf::to(dest).with(sp1_dec) (1000000000ull));
    BM(SP1_DEC, strf::to(dest).with(sp1_dec) (1000000000000000000ull));

    BM(SP2_DEC, strf::to(dest).with(sp2_dec) (1000ull));
    BM(SP2_DEC, strf::to(dest).with(sp2_dec) (1000000000ull));
    BM(SP2_DEC, strf::to(dest).with(sp2_dec) (1000000000000000000ull));

    BM(MP1_HEX, strf::to(dest).with(mp1_hex) (strf::hex(0xFFFFull)));
    BM(MP1_HEX, strf::to(dest).with(mp1_hex) (strf::hex(0xFFFFFFFFull)));
    BM(MP1_HEX, strf::to(dest).with(mp1_hex) (strf::hex(0xFFFFFFFFFFFFFFFFull)));

    BM(MP2_HEX, strf::to(dest).with(mp2_hex) (strf::hex(0xFFFFull)));
    BM(MP2_HEX, strf::to(dest).with(mp2_hex) (strf::hex(0xFFFFFFFFull)));
    BM(MP2_HEX, strf::to(dest).with(mp2_hex) (strf::hex(0xFFFFFFFFFFFFFFFFull)));

    BM(SP1_HEX, strf::to(dest).with(sp1_hex) (strf::hex(0xFFFFull)));
    BM(SP1_HEX, strf::to(dest).with(sp1_hex) (strf::hex(0xFFFFFFFFull)));
    BM(SP1_HEX, strf::to(dest).with(sp1_hex) (strf::hex(0xFFFFFFFFFFFFFFFFull)));

    BM(SP2_HEX, strf::to(dest).with(sp2_hex) (strf::hex(0xFFFFull)));
    BM(SP2_HEX, strf::to(dest).with(sp2_hex) (strf::hex(0xFFFFFFFFull)));
    BM(SP2_HEX, strf::to(dest).with(sp2_hex) (strf::hex(0xFFFFFFFFFFFFFFFFull)));

    BM(MP1_OCT, strf::to(dest).with(mp1_oct) (strf::oct(0xFFFFull)));
    BM(MP1_OCT, strf::to(dest).with(mp1_oct) (strf::oct(0xFFFFFFFFull)));
    BM(MP1_OCT, strf::to(dest).with(mp1_oct) (strf::oct(0xFFFFFFFFFFFFFFFFull)));

    BM(MP2_OCT, strf::to(dest).with(mp2_oct) (strf::oct(0xFFFFull)));
    BM(MP2_OCT, strf::to(dest).with(mp2_oct) (strf::oct(0xFFFFFFFFull)));
    BM(MP2_OCT, strf::to(dest).with(mp2_oct) (strf::oct(0xFFFFFFFFFFFFFFFFull)));

    BM(SP1_OCT, strf::to(dest).with(sp1_oct) (strf::oct(0xFFFFull)));
    BM(SP1_OCT, strf::to(dest).with(sp1_oct) (strf::oct(0xFFFFFFFFull)));
    BM(SP1_OCT, strf::to(dest).with(sp1_oct) (strf::oct(0xFFFFFFFFFFFFFFFFull)));

    BM(SP2_OCT, strf::to(dest).with(sp2_oct) (strf::oct(0xFFFFull)));
    BM(SP2_OCT, strf::to(dest).with(sp2_oct) (strf::oct(0xFFFFFFFFull)));
    BM(SP2_OCT, strf::to(dest).with(sp2_oct) (strf::oct(0xFFFFFFFFFFFFFFFFull)));

    BM(MP1_BIN, strf::to(dest).with(mp1_bin) (strf::bin(0xFFFFull)));
    BM(MP1_BIN, strf::to(dest).with(mp1_bin) (strf::bin(0xFFFFFFFFull)));
    BM(MP1_BIN, strf::to(dest).with(mp1_bin) (strf::bin(0xFFFFFFFFFFFFFFFFull)));

    BM(MP2_BIN, strf::to(dest).with(mp2_bin) (strf::bin(0xFFFFull)));
    BM(MP2_BIN, strf::to(dest).with(mp2_bin) (strf::bin(0xFFFFFFFFull)));
    BM(MP2_BIN, strf::to(dest).with(mp2_bin) (strf::bin(0xFFFFFFFFFFFFFFFFull)));

    BM(SP1_BIN, strf::to(dest).with(sp1_bin) (strf::bin(0xFFFFull)));
    BM(SP1_BIN, strf::to(dest).with(sp1_bin) (strf::bin(0xFFFFFFFFull)));
    BM(SP1_BIN, strf::to(dest).with(sp1_bin) (strf::bin(0xFFFFFFFFFFFFFFFFull)));

    BM(SP2_BIN, strf::to(dest).with(sp2_bin) (strf::bin(0xFFFFull)));
    BM(SP2_BIN, strf::to(dest).with(sp2_bin) (strf::bin(0xFFFFFFFFull)));
    BM(SP2_BIN, strf::to(dest).with(sp2_bin) (strf::bin(0xFFFFFFFFFFFFFFFFull)));

    BM(MP1_DEC, strf::to(dest).with(mp1_dec) (strf::fixed(1e+3)));
    BM(MP1_DEC, strf::to(dest).with(mp1_dec) (strf::fixed(1e+12)));
    BM(MP1_DEC, strf::to(dest).with(mp1_dec) (strf::fixed(1e+120)));

    BM(MP2_DEC, strf::to(dest).with(mp2_dec) (strf::fixed(1e+3)));
    BM(MP2_DEC, strf::to(dest).with(mp2_dec) (strf::fixed(1e+18)));
    BM(MP2_DEC, strf::to(dest).with(mp2_dec) (strf::fixed(1e+180)));

    BM(SP1_DEC, strf::to(dest).with(sp1_dec) (strf::fixed(1e+3)));
    BM(SP1_DEC, strf::to(dest).with(sp1_dec) (strf::fixed(1e+12)));
    BM(SP1_DEC, strf::to(dest).with(sp1_dec) (strf::fixed(1e+120)));

    BM(SP2_DEC, strf::to(dest).with(sp2_dec) (strf::fixed(1e+3)));
    BM(SP2_DEC, strf::to(dest).with(sp2_dec) (strf::fixed(1e+18)));
    BM(SP2_DEC, strf::to(dest).with(sp2_dec) (strf::fixed(1e+180)));

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    strf::to(stdout)(
        "\n    " STR(MP1_DEC)
        "\n    " STR(MP2_DEC)
        "\n    " STR(SP1_DEC)
        "\n    " STR(SP2_DEC)
        "\n    " STR(MP1_HEX)
        "\n    " STR(MP2_HEX)
        "\n    " STR(SP1_HEX)
        "\n    " STR(SP2_HEX)
        "\n    " STR(MP1_OCT)
        "\n    " STR(MP2_OCT)
        "\n    " STR(SP1_OCT)
        "\n    " STR(SP2_OCT)
        "\n    " STR(MP1_BIN)
        "\n    " STR(MP2_BIN)
        "\n    " STR(SP1_BIN)
        "\n    " STR(SP2_BIN) "\n" );

    return 0;
}
