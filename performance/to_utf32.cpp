//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <locale>
#include <fstream>

#include <strf.hpp>
#include "loop_timer.hpp"


int main()
{
    std::u16string u16sample1(500, u'A');
    std::u16string u16sample4;
    for(int i = 0; i < 500; ++i) u16sample4.append(u"\U00010000");

    char32_t u32output[100000];
    constexpr std::size_t u32output_size = sizeof(u32output) / sizeof(u32output[0]);
    char32_t* u32output_end = &u32output[u32output_size];

    std::cout << "\nUTF-16 to UTF-32\n";

    PRINT_BENCHMARK("format(u32output)(u16sample1)")
    {
        auto err = strf::write(u32output)(u16sample1);
        (void)err;
    }
    PRINT_BENCHMARK("format(u32output)(u16sample4)")
    {
        auto err = strf::write(u32output)(u16sample4);
        (void)err;
    }

    std::cout << "\nUTF-8 to UTF-32\n";
    std::string u8sample1(500, 'A');
    std::string u8sample2;
    std::string u8sample3;
    std::string u8sample4;
    for(int i = 0; i < 500; ++i) u8sample2.append(u8"\u0100");
    for(int i = 0; i < 500; ++i) u8sample3.append(u8"\u0800");
    for(int i = 0; i < 500; ++i) u8sample4.append(u8"\U00010000");

    PRINT_BENCHMARK("format(u32output)(u8sample1)")
    {
        auto err = strf::write(u32output)(u8sample1);
        (void)err;
    }
    PRINT_BENCHMARK("format(u32output)(u8sample2)")
    {
        auto err = strf::write(u32output)(u8sample2);
        (void)err;
    }
    PRINT_BENCHMARK("format(u32output)(u8sample3)")
    {
        auto err = strf::write(u32output)(u8sample3);
        (void)err;
    }
    PRINT_BENCHMARK("format(u32output)(u8sample4)")
    {
        auto err = strf::write(u32output)(u8sample4);
        (void)err;
    }

#if ! defined(_MSC_VER)

    std::locale::global(std::locale("en_US.utf8"));
    auto& codecvt = std::use_facet<std::codecvt<char32_t, char, std::mbstate_t>>(std::locale());
    const char* u8from_next = nullptr;
    char32_t* u32to_next = nullptr;

    std::cout << "\nUTF-8 to UTF-32 using std::codecvt<char32_t, char>\n";

    PRINT_BENCHMARK("std::codecvt / u8sample1 to utf32")
    {
        std::mbstate_t mb{};
        codecvt.in
            ( mb
            , &*u8sample1.begin()
            , &*u8sample1.end()
            , u8from_next
            , u32output
            , u32output_end
            , u32to_next);
        *u32to_next = '\0';
    }

    PRINT_BENCHMARK("std::codecvt / u8sample2 to utf32")
    {
        std::mbstate_t mb{};
        codecvt.in
            ( mb
            , &*u8sample2.begin()
            , &*u8sample2.end()
            , u8from_next
            , u32output
            , u32output_end
            , u32to_next);
        *u32to_next = '\0';
    }
    PRINT_BENCHMARK("std::codecvt / u8sample3 to utf32")
    {
        std::mbstate_t mb{};
        codecvt.in
            ( mb
            , &*u8sample3.begin()
            , &*u8sample3.end()
            , u8from_next
            , u32output
            , u32output_end
            , u32to_next);
        *u32to_next = '\0';
    }
    PRINT_BENCHMARK("std::codecvt / u8sample4 to utf32")
    {
        std::mbstate_t mb{};
        codecvt.in
            ( mb
            , &*u8sample4.begin()
            , &*u8sample4.end()
            , u8from_next
            , u32output
            , u32output_end
            , u32to_next);
        *u32to_next = '\0';
    }

#endif // ! defined(MSVC)

}
