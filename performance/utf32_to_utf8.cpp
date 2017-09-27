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


#define PRINT_BENCHMARK(label)  \
  BOOST_LOOP_TIMER(10000000000ll, label)

int main()
{
    namespace strf = boost::stringify::v0;
    
    std::u32string u32sample1(500, U'A');
    std::u32string u32sample2(500, U'\u0100');
    std::u32string u32sample3(500, U'\u0800');
    std::u32string u32sample4(500, U'\U00010000');

    char dest[100000];
    constexpr std::size_t dest_size = sizeof(dest) / sizeof(dest[0]);
    char* dest_end = &dest[dest_size];

    PRINT_BENCHMARK("write_to(dest) (u32sample1)")
    {
        strf::write_to(dest) (u32sample1);
    }
    PRINT_BENCHMARK("write_to(dest) (u32sample2)")
    {
        strf::write_to(dest) (u32sample2);
    }
    PRINT_BENCHMARK("write_to(dest) (u32sample3)")
    {
        strf::write_to(dest) (u32sample3);
    }
    PRINT_BENCHMARK("write_to(dest) (u32sample4)")
    {
        strf::write_to(dest) (u32sample4);
    }

    std::locale::global(std::locale("en_US.utf8"));
    auto& codecvt = std::use_facet<std::codecvt<char32_t, char, std::mbstate_t>>(std::locale());
    const char32_t* from_next = nullptr;
    char* to_next = nullptr;

    strf::write_to(stdout)('\n');

    PRINT_BENCHMARK("std::codecvt / u32sample1")
    {                                                 
        std::mbstate_t mb{};
        codecvt.out
            ( mb
            , &*u32sample1.begin()
            , &*u32sample1.end()
            , from_next
            , dest
            , dest_end
            , to_next);
        *to_next = '\0';
    }

    PRINT_BENCHMARK("std::codecvt / u32sample2")
    {                                                 
        std::mbstate_t mb{};
        codecvt.out
            ( mb
            , &*u32sample2.begin()
            , &*u32sample2.end()
            , from_next
            , dest
            , dest_end
            , to_next);
        *to_next = '\0';
    }
    PRINT_BENCHMARK("std::codecvt / u32sample3")
    {                                                 
        std::mbstate_t mb{};
        codecvt.out
            ( mb
            , &*u32sample3.begin()
            , &*u32sample3.end()
            , from_next
            , dest
            , dest_end
            , to_next);
        *to_next = '\0';
    }
    PRINT_BENCHMARK("std::codecvt / u32sample4")
    {                                                 
        std::mbstate_t mb{};
        codecvt.out
            ( mb
            , &*u32sample4.begin()
            , &*u32sample4.end()
            , from_next
            , dest
            , dest_end
            , to_next);
        *to_next = '\0';
    }

}
