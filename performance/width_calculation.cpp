//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <string.h>
#include <locale>
#include <fstream>
#include <codecvt>

#include <strf.hpp>
#include "loop_timer.hpp"

int main()
{
    char u8dest[100000];
    char16_t u16dest[100000];

    const std::string u8str5 {5, 'x'};
    const std::string u8str50 {50, 'x'};
    const std::u16string u16str5 {5, u'x'};
    const std::u16string u16str50 {50, u'x'};

    const auto print = strf::to(stdout);
    auto wfunc = [](char32_t ch) { return (ch == U'\u2E3A' ? 4 : ch == U'\u2014' ? 2 : 1); };
    auto custom_wcalc = strf::make_width_calculator(wfunc);

    print("UTF-8:\n");

    PRINT_BENCHMARK("strf::to(u8dest) (strf::fmt(u8str5) > 5)")
    {
        (void)strf::to(u8dest) (strf::fmt(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(strf::width_as_fast_u32len{}) (strf::fmt(u8str5) > 5)")
    {
        (void)strf::to(u8dest)
            .with(strf::width_as_fast_u32len())
            (strf::fmt(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(strf::width_as_u32len{}) (strf::fmt(u8str5) > 5)")
    {
        (void)strf::to(u8dest)
            .with(strf::width_as_u32len())
            (strf::fmt(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(custom_wcalc) (strf::fmt(u8str5) > 5)")
    {
        (void)strf::to(u8dest)
            .with(custom_wcalc)
            (strf::fmt(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u8dest) (strf::join_right(5)(u8str5))")
    {
        (void)strf::to(u8dest) (strf::join_right(5)(u8str5));
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(strf::width_as_fast_u32len{}) (strf::join_right(5)(u8str5))")
    {
        (void)strf::to(u8dest)
            .with(strf::width_as_fast_u32len())
            (strf::join_right(5)(u8str5));
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(strf::width_as_u32len{}) (strf::join_right(5)(u8str5))")
    {
        (void)strf::to(u8dest)
            .with(strf::width_as_u32len())
            (strf::join_right(5)(u8str5));
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(custom_wcalc) (strf::join_right(5)(u8str5))")
    {
        (void)strf::to(u8dest)
            .with(custom_wcalc)
            (strf::join_right(5)(u8str5));
    }

    print('\n');

    PRINT_BENCHMARK("strf::to(u8dest) (strf::fmt(u8str50) > 50)")
    {
        (void)strf::to(u8dest) (strf::fmt(u8str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(strf::width_as_fast_u32len{}) (strf::fmt(u8str50) > 50)")
    {
        (void)strf::to(u8dest)
            .with(strf::width_as_fast_u32len{})
            (strf::fmt(u8str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(strf::width_as_u32len{}) (strf::fmt(u8str50) > 50)")
    {
        (void)strf::to(u8dest)
            .with(strf::width_as_u32len{})
            (strf::fmt(u8str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(custom_wcalc) (strf::fmt(u8str50) > 50)")
    {
        (void)strf::to(u8dest)
            .with(custom_wcalc)
            (strf::fmt(u8str50) > 50);
    }

    PRINT_BENCHMARK("strf::to(u8dest) (strf::join_right(50)(u8str50))")
    {
        (void)strf::to(u8dest) (strf::join_right(50)(u8str50));
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(strf::width_as_fast_u32len{}) (strf::join_right(50)(u8str50))")
    {
        (void)strf::to(u8dest)
            .with(strf::width_as_fast_u32len())
            (strf::join_right(50)(u8str50));
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(strf::width_as_u32len{}) (strf::join_right(50)(u8str50))")
    {
        (void)strf::to(u8dest)
            .with(strf::width_as_u32len())
            (strf::join_right(50)(u8str50));
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(custom_wcalc) (strf::join_right(50)(u8str50))")
    {
        (void)strf::to(u8dest)
            .with(custom_wcalc)
            (strf::join_right(50)(u8str50));
    }

    print("\nUTF-16:\n");

    PRINT_BENCHMARK("strf::to(u16dest) (strf::fmt(u16str5) > 5)")
    {
        (void)strf::to(u16dest) (strf::fmt(u16str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_fast_u32len{}) (strf::fmt(u16str5) > 5)")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_fast_u32len{})
            (strf::fmt(u16str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_u32len{}) (strf::fmt(u16str5) > 5)")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_u32len{})
            (strf::fmt(u16str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(custom_wcalc) (strf::fmt(u16str5) > 5)")
    {
        (void)strf::to(u16dest)
            .with(custom_wcalc)
            (strf::fmt(u16str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u16dest) (strf::join_right(5)(u16str5))")
    {
        (void)strf::to(u16dest) (strf::join_right(5)(u16str5));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_fast_u32len{}) (strf::join_right(5)(u16str5))")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_fast_u32len())
            (strf::join_right(5)(u16str5));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_u32len{}) (strf::join_right(5)(u16str5))")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_u32len())
            (strf::join_right(5)(u16str5));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(custom_wcalc) (strf::join_right(5)(u16str5))")
    {
        (void)strf::to(u16dest)
            .with(custom_wcalc)
            (strf::join_right(5)(u16str5));
    }

    print('\n');
    PRINT_BENCHMARK("strf::to(u16dest) (strf::fmt(u16str50) > 50)")
    {
        (void)strf::to(u16dest) (strf::fmt(u16str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_fast_u32len{}) (strf::fmt(u16str50) > 50)")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_fast_u32len{})
            (strf::fmt(u16str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_u32len{}) (strf::fmt(u16str50) > 50)")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_u32len{})
            (strf::fmt(u16str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(custom_wcalc) (strf::fmt(u16str50) > 50)")
    {
        (void)strf::to(u16dest)
            .with(custom_wcalc)
            (strf::fmt(u16str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u16dest) (strf::join_right(50)(u16str50))")
    {
        (void)strf::to(u16dest) (strf::join_right(50)(u16str50));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_fast_u32len{}) (strf::join_right(50)(u16str50))")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_fast_u32len())
            (strf::join_right(50)(u16str50));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_u32len{}) (strf::join_right(50)(u16str50))")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_u32len())
            (strf::join_right(50)(u16str50));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(custom_wcalc) (strf::join_right(50)(u16str50))")
    {
        (void)strf::to(u16dest)
            .with(custom_wcalc)
            (strf::join_right(50)(u16str50));
    }

    print("\nWhen converting UTF-8 to UTF-16:\n");

    PRINT_BENCHMARK("strf::to(u16dest) (strf::cv(u8str5) > 5)")
    {
        (void)strf::to(u16dest) (strf::cv(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_fast_u32len{}) (strf::cv(u8str5) > 5)")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_fast_u32len())
            (strf::cv(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_u32len{}) (strf::cv(u8str5) > 5)")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_u32len())
            (strf::cv(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(custom_wcalc) (strf::cv(u8str5) > 5)")
    {
        (void)strf::to(u16dest)
            .with(custom_wcalc)
            (strf::cv(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u16dest) (strf::join_right(5)(strf::cv(u8str5)))")
    {
        (void)strf::to(u16dest) (strf::join_right(5)(strf::cv(u8str5)));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_fast_u32len{}) (strf::join_right(5)(strf::cv(u8str5)))")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_fast_u32len())
            (strf::join_right(5)(strf::cv(u8str5)));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_u32len{}) (strf::join_right(5)(strf::cv(u8str5)))")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_u32len())
            (strf::join_right(5)(strf::cv(u8str5)));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(custom_wcalc) (strf::join_right(5)(strf::cv(u8str5)))")
    {
        (void)strf::to(u16dest)
            .with(custom_wcalc)
            (strf::join_right(5)(strf::cv(u8str5)));
    }

        PRINT_BENCHMARK("strf::to(u16dest) (strf::cv(u8str50) > 50)")
    {
        (void)strf::to(u16dest) (strf::cv(u8str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_fast_u32len{}) (strf::cv(u8str50) > 50)")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_fast_u32len())
            (strf::cv(u8str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_u32len{}) (strf::cv(u8str50) > 50)")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_u32len())
            (strf::cv(u8str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(custom_wcalc) (strf::cv(u8str50) > 50)")
    {
        (void)strf::to(u16dest)
            .with(custom_wcalc)
            (strf::cv(u8str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u16dest) (strf::join_right(50)(strf::cv(u8str50)))")
    {
        (void)strf::to(u16dest) (strf::join_right(50)(strf::cv(u8str50)));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_fast_u32len{}) (strf::join_right(50)(strf::cv(u8str50)))")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_fast_u32len())
            (strf::join_right(50)(strf::cv(u8str50)));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_u32len{}) (strf::join_right(50)(strf::cv(u8str50)))")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_u32len())
            (strf::join_right(50)(strf::cv(u8str50)));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(custom_wcalc) (strf::join_right(50)(strf::cv(u8str50)))")
    {
        (void)strf::to(u16dest)
            .with(custom_wcalc)
            (strf::join_right(50)(strf::cv(u8str50)));
    }
    return 0;
}
