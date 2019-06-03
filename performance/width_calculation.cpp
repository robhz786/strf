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

//[wfunc_definition
int wfunc(int limit, const char32_t* it, const char32_t* end )
{
    int w = 0;
    for (; w < limit && it != end; ++it)
    {
        auto ch = *it;
        w += ( ch == U'\u2E3A' ? 4
             : ch == U'\u2014' ? 2
             : 1 );
    }
    return w;
}

//]

int main()
{
    namespace strf = boost::stringify::v0;

    char u8dest[100000];
    char16_t u16dest[100000];

    const std::string u8str5 {5, 'x'};
    const std::string u8str50 {50, 'x'};
    const std::u16string u16str5 {5, u'x'};
    const std::u16string u16str50 {50, u'x'};

    (void)strf::write(stdout)("UTF-8:\n");

    PRINT_BENCHMARK("strf::write(u8dest) (strf::right(u8str5, 5))")
    {
        (void)strf::write(u8dest) (strf::right(u8str5, 5));
    }
    PRINT_BENCHMARK("strf::write(u8dest) .facets(strf::width_as_u32len()) (strf::right(u8str5, 5))")
    {
        (void)strf::write(u8dest)
            .facets(strf::width_as_u32len())
            (strf::right(u8str5, 5));
    }
    PRINT_BENCHMARK("strf::write(u8dest) .facets(strf::width_as(wfunc)) (strf::right(u8str5, 5))")
    {
        (void)strf::write(u8dest)
            .facets(strf::width_as(wfunc))
            (strf::right(u8str5, 5));
    }
    std::cout << "\n";
    PRINT_BENCHMARK("strf::write(u8dest) (strf::right(u8str50, 50))")
    {
        (void)strf::write(u8dest) (strf::right(u8str50, 50));
    }
    PRINT_BENCHMARK("strf::write(u8dest) .facets(strf::width_as_u32len()) (strf::right(u8str50, 50))")
    {
        (void)strf::write(u8dest)
            .facets(strf::width_as_u32len())
            (strf::right(u8str50, 50));
    }
    PRINT_BENCHMARK("strf::write(u8dest) .facets(strf::width_as(wfunc)) (strf::right(u8str50, 50))")
    {
        (void)strf::write(u8dest)
            .facets(strf::width_as(wfunc))
            (strf::right(u8str50, 50));
    }

    (void)strf::write(stdout)("\nUTF-16:\n");

    PRINT_BENCHMARK("strf::write(u16dest) (strf::fmt_cv(u16str5) > 5)")
    {
        (void)strf::write(u16dest) (strf::fmt_cv(u16str5) > 5);
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(strf::width_as_u32len()) (strf::right(u16str5, 5))")
    {
        (void)strf::write(u16dest)
            .facets(strf::width_as_u32len())
            (strf::right(u16str5, 5));
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(strf::width_as(wfunc)) (strf::right(u16str5, 5))")
    {
        (void)strf::write(u16dest)
            .facets(strf::width_as(wfunc))
            (strf::right(u16str5, 5));
    }
    std::cout << "\n";
    PRINT_BENCHMARK("strf::write(u16dest) (strf::right(u16str50, 50))")
    {
        (void)strf::write(u16dest) (strf::right(u16str50, 50));
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(strf::width_as_u32len()) (strf::right(u16str50, 50))")
    {
        (void)strf::write(u16dest)
            .facets(strf::width_as_u32len())
            (strf::right(u16str50, 50));
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(strf::width_as(wfunc)) (strf::right(u16str50, 50))")
    {
        (void)strf::write(u16dest)
            .facets(strf::width_as(wfunc))
            (strf::right(u16str50, 50));
    }


    return 0;
}
