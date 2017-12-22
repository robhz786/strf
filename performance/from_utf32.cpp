//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <locale>
#include <fstream>

#include <boost/stringify.hpp>
#include "loop_timer.hpp"

int main()
{
    namespace strf = boost::stringify::v0;

    std::u32string u32sample1(500, U'A');
    std::u32string u32sample2(500, U'\u0100');
    std::u32string u32sample3(500, U'\u0800');
    std::u32string u32sample4(500, U'\U00010000');

    std::cout << "\nUTF-32 to UTF-16\n";

    char16_t u16dest[100000];
    constexpr std::size_t u16dest_size = sizeof(u16dest) / sizeof(u16dest[0]);
    char16_t* u16dest_end = &u16dest[u16dest_size];

    PRINT_BENCHMARK("write_to(u16dest) = {u32sample1}")
    {
        strf::write_to(u16dest) = {u32sample1};
    }
    PRINT_BENCHMARK("write_to(u16dest) = {u32sample4}")
    {
        strf::write_to(u16dest) = {u32sample4};
    }

    std::cout << "\nUTF-32 to UTF-8\n";

    char u8dest[100000];
    constexpr std::size_t u8dest_size = sizeof(u8dest) / sizeof(u8dest[0]);
    char* u8dest_end = &u8dest[u8dest_size];

    PRINT_BENCHMARK("write_to(u8dest) = {u32sample1}")
    {
        strf::write_to(u8dest) = {u32sample1};
    }
    PRINT_BENCHMARK("write_to(u8dest) = {u32sample2}")
    {
        strf::write_to(u8dest) = {u32sample2};
    }
    PRINT_BENCHMARK("write_to(u8dest) = {u32sample3}")
    {
        strf::write_to(u8dest) = {u32sample3};
    }
    PRINT_BENCHMARK("write_to(u8dest) = {u32sample4}")
    {
        strf::write_to(u8dest) = {u32sample4};
    }

#if ! defined(MSVC)

    std::locale::global(std::locale("en_US.utf8"));
    auto& codecvt = std::use_facet<std::codecvt<char32_t, char, std::mbstate_t>>(std::locale());
    const char32_t* cu32next = nullptr;
    char* u8next = nullptr;

    std::cout << "\nUTF-32 to UTF-8 using std::codecvt<char32_t, char>\n";

    PRINT_BENCHMARK("std::codecvt / u32sample1")
    {
        std::mbstate_t mb{};
        codecvt.out
            ( mb
            , &*u32sample1.begin()
            , &*u32sample1.end()
            , cu32next
            , u8dest
            , u8dest_end
            , u8next);
        *u8next = '\0';
    }

    PRINT_BENCHMARK("std::codecvt / u32sample2")
    {
        std::mbstate_t mb{};
        codecvt.out
            ( mb
            , &*u32sample2.begin()
            , &*u32sample2.end()
            , cu32next
            , u8dest
            , u8dest_end
            , u8next);
        *u8next = '\0';
    }
    PRINT_BENCHMARK("std::codecvt / u32sample3")
    {
        std::mbstate_t mb{};
        codecvt.out
            ( mb
            , &*u32sample3.begin()
            , &*u32sample3.end()
            , cu32next
            , u8dest
            , u8dest_end
            , u8next);
        *u8next = '\0';
    }
    PRINT_BENCHMARK("std::codecvt / u32sample4")
    {
        std::mbstate_t mb{};
        codecvt.out
            ( mb
            , &*u32sample4.begin()
            , &*u32sample4.end()
            , cu32next
            , u8dest
            , u8dest_end
            , u8next);
        *u8next = '\0';
    }

#endif // ! defined(MSVC)

}
