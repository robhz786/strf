//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <locale>
#include <codecvt>
#include <fstream>
#include <benchmark/benchmark.h>
#include <strf.hpp>

static void fill_with_codepoints
    ( std::integral_constant<std::size_t, 1>
    , char16_t* str
    , std::uint32_t count)
{
    for(std::uint32_t i = 0, x = 1; i < count; ++i, ++x){
        if (x >= 0x80) {
            x = 0;
        }
        str[i] = static_cast<char16_t>(x);
    }
}
static void fill_with_codepoints
    ( std::integral_constant<std::size_t, 2>
    , char16_t* str
    , std::uint32_t count)
{
    auto count2 = 2 * count;
    for(std::uint32_t i = 0, x = 0x80; i < count2; ++i, x += 0x10){
        if (x >= 0x800) {
            x = 0x80;
        }
        str[i] = static_cast<char16_t>(x);
    }
}

static void fill_with_codepoints
    ( std::integral_constant<std::size_t, 3>
    , char16_t* str
    , std::uint32_t count)
{
    auto count3 = 3 * count;
    for(std::uint32_t i = 0, x = 0x800; i < count3; ++ i, x += 0x100){
        if (x >= 0xD800 && x <= 0xDFFF){
            x= 0xE000;
        }
        if (x >= 0x10000) {
            x = 0x800;
        }
        str[i] = static_cast<char16_t>(x);
    }
}

static void fill_with_codepoints
    ( std::integral_constant<std::size_t, 4>
    , char16_t* str
    , std::uint32_t count)
{
    auto count4 = 4 * count;
    for(std::uint32_t i = 0, x = 0x10000; i < count4; i += 2, x += 0x1000){
        if (x >= 0x10FFFF) {
            x = 0x10000;
        }
        x -= 0x10000;
        str[i]     = static_cast<char16_t>(0xD800 | (x >> 10));
        str[i + 1] = static_cast<char16_t>(0xDC00 | (x & 0x3FF));
    }
}

template <std::size_t CodepointsSize>
void fill_with_codepoints(char16_t* str, unsigned count) {
    fill_with_codepoints(std::integral_constant<std::size_t, CodepointsSize>(), str, count);
}

template <std::size_t CodepointsCount, std::size_t CodepointsSize>
static void bm_strf(benchmark::State& state) {
    char16_t u16sample[CodepointsCount * 2];
    char     u8dest   [CodepointsCount * 4 + 1];
    fill_with_codepoints<CodepointsSize>(u16sample, CodepointsCount);
    strf::detail::simple_string_view<char16_t> u16str
        (u16sample, CodepointsCount * (1 + (CodepointsSize==4)));
    for(auto _ : state) {
        strf::to(u8dest)(strf::conv(u16str));
        benchmark::DoNotOptimize(u8dest);
    }
}

template <std::size_t CodepointsCount, std::size_t CodepointsSize>
static void bm_codecvt(benchmark::State& state) {
    char16_t u16sample[CodepointsCount * 2];
    char     u8dest   [CodepointsCount * 4 + 1];
    const char16_t* u16from_next = nullptr;
    char* u8to_next = nullptr;
    fill_with_codepoints<CodepointsSize>(u16sample, CodepointsCount);
    std::codecvt_utf8_utf16<char16_t> codecvt;
    for(auto _ : state) {
        std::mbstate_t mb{};
        codecvt.out
            ( mb
            , u16sample
            , u16sample + CodepointsCount * (1 + (CodepointsSize == 4))
            , u16from_next
            , u8dest
            , u8dest + CodepointsCount * 4
            , u8to_next);
        benchmark::DoNotOptimize(u8dest);
    }
}

static void dummy (benchmark::State&)
{
}

int main(int argc, char** argv)
{
    benchmark::RegisterBenchmark("strf::to(u8dest)(strf::conv(u16small1))", bm_strf<20, 1>);
    benchmark::RegisterBenchmark("strf::to(u8dest)(strf::conv(u16small2))", bm_strf<20, 2>);
    benchmark::RegisterBenchmark("strf::to(u8dest)(strf::conv(u16small3))", bm_strf<20, 3>);
    benchmark::RegisterBenchmark("strf::to(u8dest)(strf::conv(u16small4))", bm_strf<20, 4>);

    benchmark::RegisterBenchmark("    -------------", dummy);

    benchmark::RegisterBenchmark("strf::to(u8dest)(strf::conv(u16big1))", bm_strf<200, 1>);
    benchmark::RegisterBenchmark("strf::to(u8dest)(strf::conv(u16big2))", bm_strf<200, 2>);
    benchmark::RegisterBenchmark("strf::to(u8dest)(strf::conv(u16big3))", bm_strf<200, 3>);
    benchmark::RegisterBenchmark("strf::to(u8dest)(strf::conv(u16big4))", bm_strf<200, 4>);

    benchmark::RegisterBenchmark("    -------------", dummy);

    benchmark::RegisterBenchmark("std::codecvt / u16small1 to utf8", bm_codecvt<20, 1>);
    benchmark::RegisterBenchmark("std::codecvt / u16small2 to utf8", bm_codecvt<20, 2>);
    benchmark::RegisterBenchmark("std::codecvt / u16small3 to utf8", bm_codecvt<20, 3>);
    benchmark::RegisterBenchmark("std::codecvt / u16small4 to utf8", bm_codecvt<20, 4>);

    benchmark::RegisterBenchmark("    -------------", dummy);

    benchmark::RegisterBenchmark("std::codecvt / u16big1 to utf8", bm_codecvt<200, 1>);
    benchmark::RegisterBenchmark("std::codecvt / u16big2 to utf8", bm_codecvt<200, 2>);
    benchmark::RegisterBenchmark("std::codecvt / u16big3 to utf8", bm_codecvt<200, 3>);
    benchmark::RegisterBenchmark("std::codecvt / u16big4 to utf8", bm_codecvt<200, 4>);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
