//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <locale>
#include <fstream>
#include <codecvt>

#include <boost/stringify.hpp>
#include "loop_timer.hpp"


int main()
{
    namespace strf = boost::stringify::v0;
    
    char u8dest[100000];
    //constexpr std::size_t u8dest_size = sizeof(u8dest) / sizeof(u8dest[0]);
    //char* u8dest_end = &u8dest[u8dest_size];

    char16_t u16dest[100000];
    //constexpr std::size_t u16dest_size = sizeof(u16dest) / sizeof(u16dest[0]);
    //char16_t* u16dest_end = &u16dest[u16dest_size];

    (void)strf::format(stdout)("UTF-8:\n");
    
    PRINT_BENCHMARK("format(u8dest) .facets(strf::width_as_codepoints_count())(\"aaaaa\", strf::right(u8\"bbb\\03B1\\03B2\", 10))")
    {
        (void)strf::format(u8dest)
            .facets(strf::width_as_codepoints_count())
            ("aaaaa", strf::right(u8"bbb\03B1\03B2", 10));
    }
    PRINT_BENCHMARK("format(u8dest)(\"aaaaa\", strf::right(u8\"bbb\\03B1\\03B2\", 10))")
    {
        (void)strf::format(u8dest)("aaa", strf::right(u8"bbb\03B1\03B2", 10));
    }
    PRINT_BENCHMARK("format(u8dest) .facets(strf::width_as_codepoints_count())(strf::right(u8str_50, 60))")
    {
        (void)strf::format(u8dest).facets(strf::width_as_codepoints_count())
           (strf::right("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 60));
    }
    PRINT_BENCHMARK("format(u8dest) ={strf::right(u8str_50, 60)}")
    {
        (void)strf::format(u8dest)(strf::right("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 60));
    }

    (void)strf::format(stdout)("\nUTF-8:\n");
    
    PRINT_BENCHMARK("format(u16dest) .facets(strf::width_as_codepoints_count())(u\"aaaaa\", strf::right(u\"bbb\\03B1\\03B2\", 10))")
    {
        (void)strf::format(u16dest).facets(strf::width_as_codepoints_count())
           (u"aaaaa", strf::right(u"bbb\03B1\03B2", 10));

    }
    PRINT_BENCHMARK("format(u16dest)(u\"aaaaa\", strf::right(u\"bbb\\03B1\\03B2\", 10))")
    {
        (void)strf::format(u16dest)(u"aaaaa", strf::right(u"bbb\03B1\03B2", 10));
    }
    PRINT_BENCHMARK("format(u16dest) .facets(strf::width_as_codepoints_count())(strf::right(u16str_50, 60))")
    {
        (void)strf::format(u16dest).facets(strf::width_as_codepoints_count())
            (strf::right(u"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 60));
    }
    PRINT_BENCHMARK("format(u16dest) ={strf::right(u16str_50, 60)}")
    {
        (void)strf::format(u16dest)
            (strf::right(u"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 60));
    }



    
    return 0;
}
