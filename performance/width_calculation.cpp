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

    strf::format(stdout).exception("UTF-8:\n");
    
    PRINT_BENCHMARK("format(u8dest) .facets(strf::width_as_codepoints_count()).exception(\"aaaaa\", strf::right(u8\"bbb\\03B1\\03B2\", 10))")
    {
        strf::format(u8dest)
            .facets(strf::width_as_codepoints_count())
            .exception("aaaaa", strf::right(u8"bbb\03B1\03B2", 10));
    }
    PRINT_BENCHMARK("format(u8dest).exception(\"aaaaa\", strf::right(u8\"bbb\\03B1\\03B2\", 10))")
    {
        strf::format(u8dest).exception("aaa", strf::right(u8"bbb\03B1\03B2", 10));
    }
    PRINT_BENCHMARK("format(u8dest) .facets(strf::width_as_codepoints_count()).exception(strf::right(u8str_50, 60))")
    {
        strf::format(u8dest).facets(strf::width_as_codepoints_count())
           .exception(strf::right("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 60));
    }
    PRINT_BENCHMARK("format(u8dest) ={strf::right(u8str_50, 60)}")
    {
        strf::format(u8dest).exception(strf::right("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 60));
    }

    strf::format(stdout).exception("\nUTF-8:\n");
    
    PRINT_BENCHMARK("format(u16dest) .facets(strf::width_as_codepoints_count()).exception(u\"aaaaa\", strf::right(u\"bbb\\03B1\\03B2\", 10))")
    {
        strf::format(u16dest).facets(strf::width_as_codepoints_count())
           .exception(u"aaaaa", strf::right(u"bbb\03B1\03B2", 10));

    }
    PRINT_BENCHMARK("format(u16dest).exception(u\"aaaaa\", strf::right(u\"bbb\\03B1\\03B2\", 10))")
    {
        strf::format(u16dest).exception(u"aaaaa", strf::right(u"bbb\03B1\03B2", 10));
    }
    PRINT_BENCHMARK("format(u16dest) .facets(strf::width_as_codepoints_count()).exception(strf::right(u16str_50, 60))")
    {
        strf::format(u16dest).facets(strf::width_as_codepoints_count())
            .exception(strf::right(u"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 60));
    }
    PRINT_BENCHMARK("format(u16dest) ={strf::right(u16str_50, 60)}")
    {
        strf::format(u16dest)
            .exception(strf::right(u"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 60));
    }



    
    return 0;
}
