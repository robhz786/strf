//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <benchmark/benchmark.h>
#include <strf.hpp>

#define CREATE_BENCHMARK(PREFIX)                             \
    static void PREFIX ## _func (benchmark::State& state) {  \
        char      u8dest[110];                               \
        char16_t u16dest[110];                               \
        const std::string u8str5 {5, 'x'};                   \
        const std::string u8str50 {50, 'x'};                 \
        const std::u16string u16str5 {5, u'x'};              \
        const std::u16string u16str50 {50, u'x'};            \
        (void) u8dest;                                       \
        (void) u16dest;                                      \
        (void) u8str5;                                       \
        (void) u8str50;                                      \
        (void) u16str5;                                      \
        (void) u16str50;                                     \
        for(auto _ : state) {                                \
            PREFIX ## _OP ;                                  \
            benchmark::DoNotOptimize(u8dest);                \
            benchmark::DoNotOptimize(u16dest);               \
            benchmark::ClobberMemory();                      \
        }                                                    \
    }

#define REGISTER_BENCHMARK(X) benchmark::RegisterBenchmark(STR(X ## _OP), X ## _func);
#define STR2(X) #X
#define STR(X) STR2(X)


constexpr auto fast_width  = strf::fast_width();
constexpr auto slow_u32len = strf::width_as_fast_u32len();
constexpr auto fast_u32len = strf::width_as_u32len();
auto wfunc = [](char32_t ch) { return (ch == U'\u2E3A' ? 4 : ch == U'\u2014' ? 2 : 1); };
const auto custom_calc = strf::make_width_calculator(wfunc);


#define  U8_F_5_OP      strf::to(u8dest) .with( fast_width) (strf::fmt(u8str5) > 5);
#define  U8_F32L_5_OP   strf::to(u8dest) .with(fast_u32len) (strf::fmt(u8str5) > 5);
#define  U8_32L_5_OP    strf::to(u8dest) .with(slow_u32len) (strf::fmt(u8str5) > 5);
#define  U8_C_5_OP      strf::to(u8dest) .with(custom_calc) (strf::fmt(u8str5) > 5);

#define  U8_F_J5_OP     strf::to(u8dest) .with( fast_width) (strf::join_right(5)(u8str5));
#define  U8_F32L_J5_OP  strf::to(u8dest) .with(fast_u32len) (strf::join_right(5)(u8str5));
#define  U8_32L_J5_OP   strf::to(u8dest) .with(slow_u32len) (strf::join_right(5)(u8str5));
#define  U8_C_J5_OP     strf::to(u8dest) .with(custom_calc) (strf::join_right(5)(u8str5));

#define  U8_F_50_OP     strf::to(u8dest) .with( fast_width) (strf::fmt(u8str50) > 50);
#define  U8_F32L_50_OP  strf::to(u8dest) .with(fast_u32len) (strf::fmt(u8str50) > 50);
#define  U8_32L_50_OP   strf::to(u8dest) .with(slow_u32len) (strf::fmt(u8str50) > 50);
#define  U8_C_50_OP     strf::to(u8dest) .with(custom_calc) (strf::fmt(u8str50) > 50);

#define  U8_F_J50_OP    strf::to(u8dest) .with( fast_width) (strf::join_right(50)(u8str50));
#define  U8_F32L_J50_OP strf::to(u8dest) .with(fast_u32len) (strf::join_right(50)(u8str50));
#define  U8_32L_J50_OP  strf::to(u8dest) .with(slow_u32len) (strf::join_right(50)(u8str50));
#define  U8_C_J50_OP    strf::to(u8dest) .with(custom_calc) (strf::join_right(50)(u8str50));

#define U16_F_5_OP      strf::to(u16dest).with( fast_width) (strf::fmt(u16str5) > 5);
#define U16_F32L_5_OP   strf::to(u16dest).with(fast_u32len) (strf::fmt(u16str5) > 5);
#define U16_32L_5_OP    strf::to(u16dest).with(slow_u32len) (strf::fmt(u16str5) > 5);
#define U16_C_5_OP      strf::to(u16dest).with(custom_calc) (strf::fmt(u16str5) > 5);

#define U16_F_J5_OP     strf::to(u16dest).with( fast_width) (strf::join_right(5)(u16str5));
#define U16_F32L_J5_OP  strf::to(u16dest).with(fast_u32len) (strf::join_right(5)(u16str5));
#define U16_32L_J5_OP   strf::to(u16dest).with(slow_u32len) (strf::join_right(5)(u16str5));
#define U16_C_J5_OP     strf::to(u16dest).with(custom_calc) (strf::join_right(5)(u16str5));

#define U16_F_50_OP     strf::to(u16dest).with( fast_width) (strf::fmt(u16str50) > 50);
#define U16_F32L_50_OP  strf::to(u16dest).with(fast_u32len) (strf::fmt(u16str50) > 50);
#define U16_32L_50_OP   strf::to(u16dest).with(slow_u32len) (strf::fmt(u16str50) > 50);
#define U16_C_50_OP     strf::to(u16dest).with(custom_calc) (strf::fmt(u16str50) > 50);

#define U16_F_J50_OP    strf::to(u16dest).with( fast_width) (strf::join_right(50)(u16str50));
#define U16_F32L_J50_OP strf::to(u16dest).with(fast_u32len) (strf::join_right(50)(u16str50));
#define U16_32L_J50_OP  strf::to(u16dest).with(slow_u32len) (strf::join_right(50)(u16str50));
#define U16_C_J50_OP    strf::to(u16dest).with(custom_calc) (strf::join_right(50)(u16str50));

#define  CV_F_5_OP      strf::to(u16dest).with( fast_width) (strf::cv(u8str5) > 5);
#define  CV_F32L_5_OP   strf::to(u16dest).with(fast_u32len) (strf::cv(u8str5) > 5);
#define  CV_32L_5_OP    strf::to(u16dest).with(slow_u32len) (strf::cv(u8str5) > 5);
#define  CV_C_5_OP      strf::to(u16dest).with(custom_calc) (strf::cv(u8str5) > 5);

#define  CV_F_J5_OP     strf::to(u16dest).with( fast_width) (strf::join_right(5)(strf::cv(u8str5)));
#define  CV_F32L_J5_OP  strf::to(u16dest).with(fast_u32len) (strf::join_right(5)(strf::cv(u8str5)));
#define  CV_32L_J5_OP   strf::to(u16dest).with(slow_u32len) (strf::join_right(5)(strf::cv(u8str5)));
#define  CV_C_J5_OP     strf::to(u16dest).with(custom_calc) (strf::join_right(5)(strf::cv(u8str5)));

#define  CV_F_50_OP     strf::to(u16dest).with( fast_width) (strf::cv(u8str50) > 50);
#define  CV_F32L_50_OP  strf::to(u16dest).with(fast_u32len) (strf::cv(u8str50) > 50);
#define  CV_32L_50_OP   strf::to(u16dest).with(slow_u32len) (strf::cv(u8str50) > 50);
#define  CV_C_50_OP     strf::to(u16dest).with(custom_calc) (strf::cv(u8str50) > 50);

#define  CV_F_J50_OP    strf::to(u16dest).with( fast_width) (strf::join_right(50)(strf::cv(u8str50)));
#define  CV_F32L_J50_OP strf::to(u16dest).with(fast_u32len) (strf::join_right(50)(strf::cv(u8str50)));
#define  CV_32L_J50_OP  strf::to(u16dest).with(slow_u32len) (strf::join_right(50)(strf::cv(u8str50)));
#define  CV_C_J50_OP    strf::to(u16dest).with(custom_calc) (strf::join_right(50)(strf::cv(u8str50)));


CREATE_BENCHMARK( U8_F_5);
CREATE_BENCHMARK( U8_F32L_5);
CREATE_BENCHMARK( U8_32L_5);
CREATE_BENCHMARK( U8_C_5);
CREATE_BENCHMARK( U8_F_J5);
CREATE_BENCHMARK( U8_F32L_J5);
CREATE_BENCHMARK( U8_32L_J5);
CREATE_BENCHMARK( U8_C_J5);
CREATE_BENCHMARK( U8_F_50);
CREATE_BENCHMARK( U8_F32L_50);
CREATE_BENCHMARK( U8_32L_50);
CREATE_BENCHMARK( U8_C_50);
CREATE_BENCHMARK( U8_F_J50);
CREATE_BENCHMARK( U8_F32L_J50);
CREATE_BENCHMARK( U8_32L_J50);
CREATE_BENCHMARK( U8_C_J50);
CREATE_BENCHMARK(U16_F_5);
CREATE_BENCHMARK(U16_F32L_5);
CREATE_BENCHMARK(U16_32L_5);
CREATE_BENCHMARK(U16_C_5);
CREATE_BENCHMARK(U16_F_J5);
CREATE_BENCHMARK(U16_F32L_J5);
CREATE_BENCHMARK(U16_32L_J5);
CREATE_BENCHMARK(U16_C_J5);
CREATE_BENCHMARK(U16_F_50);
CREATE_BENCHMARK(U16_F32L_50);
CREATE_BENCHMARK(U16_32L_50);
CREATE_BENCHMARK(U16_C_50);
CREATE_BENCHMARK(U16_F_J50);
CREATE_BENCHMARK(U16_F32L_J50);
CREATE_BENCHMARK(U16_32L_J50);
CREATE_BENCHMARK(U16_C_J50);
CREATE_BENCHMARK( CV_F_5);
CREATE_BENCHMARK( CV_F32L_5);
CREATE_BENCHMARK( CV_32L_5);
CREATE_BENCHMARK( CV_C_5);
CREATE_BENCHMARK( CV_F_J5);
CREATE_BENCHMARK( CV_F32L_J5);
CREATE_BENCHMARK( CV_32L_J5);
CREATE_BENCHMARK( CV_C_J5);
CREATE_BENCHMARK( CV_F_50);
CREATE_BENCHMARK( CV_F32L_50);
CREATE_BENCHMARK( CV_32L_50);
CREATE_BENCHMARK( CV_C_50);
CREATE_BENCHMARK( CV_F_J50);
CREATE_BENCHMARK( CV_F32L_J50);
CREATE_BENCHMARK( CV_32L_J50);
CREATE_BENCHMARK( CV_C_J50);

int main(int argc, char** argv)
{
    REGISTER_BENCHMARK( U8_F_5);
    REGISTER_BENCHMARK( U8_F32L_5);
    REGISTER_BENCHMARK( U8_32L_5);
    REGISTER_BENCHMARK( U8_C_5);
    REGISTER_BENCHMARK( U8_F_J5);
    REGISTER_BENCHMARK( U8_F32L_J5);
    REGISTER_BENCHMARK( U8_32L_J5);
    REGISTER_BENCHMARK( U8_C_J5);
    REGISTER_BENCHMARK( U8_F_50);
    REGISTER_BENCHMARK( U8_F32L_50);
    REGISTER_BENCHMARK( U8_32L_50);
    REGISTER_BENCHMARK( U8_C_50);
    REGISTER_BENCHMARK( U8_F_J50);
    REGISTER_BENCHMARK( U8_F32L_J50);
    REGISTER_BENCHMARK( U8_32L_J50);
    REGISTER_BENCHMARK( U8_C_J50);
    REGISTER_BENCHMARK(U16_F_5);
    REGISTER_BENCHMARK(U16_F32L_5);
    REGISTER_BENCHMARK(U16_32L_5);
    REGISTER_BENCHMARK(U16_C_5);
    REGISTER_BENCHMARK(U16_F_J5);
    REGISTER_BENCHMARK(U16_F32L_J5);
    REGISTER_BENCHMARK(U16_32L_J5);
    REGISTER_BENCHMARK(U16_C_J5);
    REGISTER_BENCHMARK(U16_F_50);
    REGISTER_BENCHMARK(U16_F32L_50);
    REGISTER_BENCHMARK(U16_32L_50);
    REGISTER_BENCHMARK(U16_C_50);
    REGISTER_BENCHMARK(U16_F_J50);
    REGISTER_BENCHMARK(U16_F32L_J50);
    REGISTER_BENCHMARK(U16_32L_J50);
    REGISTER_BENCHMARK(U16_C_J50);
    REGISTER_BENCHMARK( CV_F_5);
    REGISTER_BENCHMARK( CV_F32L_5);
    REGISTER_BENCHMARK( CV_32L_5);
    REGISTER_BENCHMARK( CV_C_5);
    REGISTER_BENCHMARK( CV_F_J5);
    REGISTER_BENCHMARK( CV_F32L_J5);
    REGISTER_BENCHMARK( CV_32L_J5);
    REGISTER_BENCHMARK( CV_C_J5);
    REGISTER_BENCHMARK( CV_F_50);
    REGISTER_BENCHMARK( CV_F32L_50);
    REGISTER_BENCHMARK( CV_32L_50);
    REGISTER_BENCHMARK( CV_C_50);
    REGISTER_BENCHMARK( CV_F_J50);
    REGISTER_BENCHMARK( CV_F32L_J50);
    REGISTER_BENCHMARK( CV_32L_J50);
    REGISTER_BENCHMARK( CV_C_J50);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    //strf::to(stdout)();

    return 0;
}
