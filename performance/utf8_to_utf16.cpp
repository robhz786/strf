//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <locale>
#include <fstream>
#include <codecvt>
#include <benchmark/benchmark.h>
#include <strf.hpp>

#if defined(_MSC_VER)
// disable warning that std::codecvt_utf8_utf16 is deprecated
#pragma warning (disable:4996)
#endif

static void fill_with_codepoints
    ( std::integral_constant<std::size_t, 1>
    , char* str
    , unsigned count)
{
    for(unsigned i = 0, x = 1; i < count; ++i, ++x){
        if (x >= 0x80) {
            x = 0;
        }
        str[i] = static_cast<char>(x);
    }
}
static void fill_with_codepoints
    ( std::integral_constant<std::size_t, 2>
    , char* str
    , unsigned count)
{
    auto count2 = 2 * count;
    for(unsigned i = 0, x = 0x80; i < count2; i += 2, x += 0x10){
        if (x >= 0x800) {
            x = 0x80;
        }
        str[i]     = static_cast<char>(0xC0 | ( x >> 6 ));
        str[i + 1] = static_cast<char>(0x80 | ( x & 0x3F ));
    }
}
static void fill_with_codepoints
    ( std::integral_constant<std::size_t, 3>
    , char* str
    , unsigned count)
{
    auto count3 = 3 * count;
    for(unsigned i = 0, x = 0x800; i < count3; i += 3, x += 0x100){
        if (x >= 0xD800 && x <= 0xDFFF){
            x= 0xE000;
        }
        if (x >= 0x10000) {
            x = 0x800;
        }
        str[i]     = static_cast<char>(0xE0 | ( x >> 12 ));
        str[i + 1] = static_cast<char>(0x80 | ( (x >> 6) & 0x3F ));
        str[i + 2] = static_cast<char>(0x80 | ( x & 0x3F ));
    }
}
static void fill_with_codepoints
    ( std::integral_constant<std::size_t, 4>
    , char* str
    , unsigned count)
{
    auto count4 = 4 * count;
    for(unsigned i = 0, x = 0x10000; i < count4; i += 4, x += 0x1000){
        if (x >= 0x10FFFF) {
            x = 0x10000;
        }
        str[i]     = static_cast<char>(0xF0 | ( x >> 18 ));
        str[i + 1] = static_cast<char>(0x80 | ( (x >> 12) & 0x3F ));
        str[i + 2] = static_cast<char>(0x80 | ( (x >> 6) & 0x3F ));
        str[i + 3] = static_cast<char>(0x80 | ( x & 0x3F ));
    }
}

template <std::size_t CodepointsSize>
void fill_with_codepoints(char* str, unsigned count) {
    fill_with_codepoints(std::integral_constant<std::size_t, CodepointsSize>(), str, count);
}

template <std::size_t CodepointsCount, std::size_t CodepointsSize>
static void bm_strf(benchmark::State& state) {
    constexpr std::size_t u8buff_size = CodepointsCount * 4;
    char u8sample[u8buff_size];
    char16_t u16dest[CodepointsCount * 2 + 1];
    fill_with_codepoints<CodepointsSize>(u8sample, CodepointsCount);
    strf::detail::simple_string_view<char> u8str(u8sample, CodepointsCount * CodepointsSize);
    for(auto _ : state) {
        strf::to(u16dest)(strf::conv(u8str));
        benchmark::DoNotOptimize(u16dest);
        //benchmark::DoNotOptimize(u8sample);
    }
}

template <std::size_t CodepointsCount, std::size_t CodepointsSize>
static void bm_codecvt(benchmark::State& state) {
    constexpr std::size_t u8buff_size = CodepointsCount * 4;
    constexpr std::size_t u16buff_size = u8buff_size / 2;
    char u8sample[u8buff_size];
    char16_t u16dest[CodepointsCount * 2 + 1];
    const char* u8from_next = nullptr;
    char16_t* u16to_next = nullptr;
    fill_with_codepoints<CodepointsSize>(u8sample, CodepointsCount);
    strf::detail::simple_string_view<char> u8str(u8sample, CodepointsCount * CodepointsSize);
    std::codecvt_utf8_utf16<char16_t> codecvt;
    for(auto _ : state) {
        std::mbstate_t mb{};
        codecvt.in
            ( mb
            , u8sample
            , u8sample + u8buff_size
            , u8from_next
            , u16dest
            , u16dest + u16buff_size
            , u16to_next);
        *u16to_next = '\0';
        benchmark::DoNotOptimize(u16dest);
    }
}

static void dummy (benchmark::State&)
{
}

int main(int argc, char** argv)
{
    benchmark::RegisterBenchmark("strf::to(u16dest)(strf::conv(u8small1))", bm_strf<20, 1>);
    benchmark::RegisterBenchmark("strf::to(u16dest)(strf::conv(u8small2))", bm_strf<20, 2>);
    benchmark::RegisterBenchmark("strf::to(u16dest)(strf::conv(u8small3))", bm_strf<20, 3>);
    benchmark::RegisterBenchmark("strf::to(u16dest)(strf::conv(u8small4))", bm_strf<20, 4>);

    benchmark::RegisterBenchmark("    -------------", dummy);

    benchmark::RegisterBenchmark("strf::to(u16dest)(strf::conv(u8big1))", bm_strf<200, 1>);
    benchmark::RegisterBenchmark("strf::to(u16dest)(strf::conv(u8big2))", bm_strf<200, 2>);
    benchmark::RegisterBenchmark("strf::to(u16dest)(strf::conv(u8big3))", bm_strf<200, 3>);
    benchmark::RegisterBenchmark("strf::to(u16dest)(strf::conv(u8big4))", bm_strf<200, 4>);

    benchmark::RegisterBenchmark("    -------------", dummy);

    benchmark::RegisterBenchmark("std::codecvt / u8small1 to utf16", bm_codecvt<20, 1>);
    benchmark::RegisterBenchmark("std::codecvt / u8small2 to utf16", bm_codecvt<20, 2>);
    benchmark::RegisterBenchmark("std::codecvt / u8small3 to utf16", bm_codecvt<20, 3>);
    benchmark::RegisterBenchmark("std::codecvt / u8small4 to utf16", bm_codecvt<20, 4>);

    benchmark::RegisterBenchmark("    -------------", dummy);

    benchmark::RegisterBenchmark("std::codecvt / u8big1 to utf16", bm_codecvt<200, 1>);
    benchmark::RegisterBenchmark("std::codecvt / u8big2 to utf16", bm_codecvt<200, 2>);
    benchmark::RegisterBenchmark("std::codecvt / u8big3 to utf16", bm_codecvt<200, 3>);
    benchmark::RegisterBenchmark("std::codecvt / u8big4 to utf16", bm_codecvt<200, 4>);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}

