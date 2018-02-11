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

    strf::write_to(stdout) &= {"UTF-8:\n"};
    
    PRINT_BENCHMARK("write_to(u8dest) .with(strf::width_as_codepoints_count()) &= {\"aaaaa\", strf::right(u8\"bbb\\03B1\\03B2\", 10)}")
    {
        strf::write_to(u8dest).with(strf::width_as_codepoints_count())
            &= {"aaaaa", strf::right(u8"bbb\03B1\03B2", 10)};
    }
    PRINT_BENCHMARK("write_to(u8dest) &= {\"aaaaa\", strf::right(u8\"bbb\\03B1\\03B2\", 10)}")
    {
        strf::write_to(u8dest) &= {"aaa", strf::right(u8"bbb\03B1\03B2", 10)};
    }
    PRINT_BENCHMARK("write_to(u8dest) .with(strf::width_as_codepoints_count()) &= {strf::right(u8str_50, 60)}")
    {
        strf::write_to(u8dest).with(strf::width_as_codepoints_count())
            &= {strf::right("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 60)};
    }
    PRINT_BENCHMARK("write_to(u8dest) ={strf::right(u8str_50, 60)}")
    {
        strf::write_to(u8dest)
         &= {strf::right("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 60)};
    }

    strf::write_to(stdout) &= {"\nUTF-8:\n"};
    
    PRINT_BENCHMARK("write_to(u16dest) .with(strf::width_as_codepoints_count()) &= {u\"aaaaa\", strf::right(u\"bbb\\03B1\\03B2\", 10)}")
    {
        strf::write_to(u16dest).with(strf::width_as_codepoints_count())
            &= {u"aaaaa", strf::right(u"bbb\03B1\03B2", 10)};

    }
    PRINT_BENCHMARK("write_to(u16dest) &= {u\"aaaaa\", strf::right(u\"bbb\\03B1\\03B2\", 10)}")
    {
        strf::write_to(u16dest) &= {u"aaaaa", strf::right(u"bbb\03B1\03B2", 10)};
    }
    PRINT_BENCHMARK("write_to(u16dest) .with(strf::width_as_codepoints_count()) &= {strf::right(u16str_50, 60)}")
    {
        strf::write_to(u16dest).with(strf::width_as_codepoints_count())
         &= {strf::right(u"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 60)};
    }
    PRINT_BENCHMARK("write_to(u16dest) ={strf::right(u16str_50, 60)}")
    {
        strf::write_to(u16dest)
         &= {strf::right(u"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 60)};
    }



    
    return 0;
}
