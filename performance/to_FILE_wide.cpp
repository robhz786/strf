//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <climits>
#include <boost/stringify.hpp>
#include "loop_timer.hpp"
#include "fmt/format.h"


#define PRINT_BENCHMARK(label)  \
  BOOST_LOOP_TIMER(10000000000ll, label)

int main()
{
    namespace strf = boost::stringify;

    FILE* dest = fopen("/dev/null", "w");
    std::size_t count;


    std::cout << "\n small strings \n";

    PRINT_BENCHMARK("strf::wwrite(dest) [{L\"Hello \", L\"World\", L\"!\"}]")
    {
        strf::wwrite(dest) [{L"Hello ", L"World", L"!"}];
    }
    PRINT_BENCHMARK("strf::wwrite(dest) (L\"Hello {}!\") = {L\"World\"}")
    {
        strf::wwrite(dest) (L"Hello {}!") = {L"World"};
    }
    PRINT_BENCHMARK("fwprintf(dest, L\"Hello %s!\", L\"World\")")
    {
        fwprintf(dest, L"Hello %s!", L"World");
    }

    std::cout << "\n long string ( 1000 characters ): \n";

    {
        std::string std_string_long_string(1000, 'x');
        const char* long_string = std_string_long_string.c_str();

        PRINT_BENCHMARK("strf::wwrite(dest) [{L\"Hello \", long_string, L\"!\"}]")
        {
            strf::wwrite(dest) [{L"Hello ", long_string, L"!"}];
        }
        PRINT_BENCHMARK("strf::wwrite(dest) (L\"Hello {}!\") = {long_string}")
        {
            strf::wwrite(dest) (L"Hello {}!") = {long_string};
        }
         PRINT_BENCHMARK("fwprintf(dest, L\"Hello %s!\", long_string)")
        {
            fwprintf(dest, L"Hello %s!", long_string);
        }
    }

    std::cout << "\n integers \n";

    PRINT_BENCHMARK("strf::wwrite(dest) [{25}]")
    {
        strf::wwrite(dest) [{25}];
    }
    PRINT_BENCHMARK("fwprintf(dest, L\"%d\", 25)")
    {
        fwprintf(dest, L"%d", 25);
    }

    std::cout << std::endl;
    PRINT_BENCHMARK("strf::wwrite(dest) [{INT_MAX}]")
    {
        strf::wwrite(dest) [{INT_MAX}];
    }
    PRINT_BENCHMARK("fwprintf(dest, L\"%d\", INT_MAX)")
    {
        fwprintf(dest, L"%d", INT_MAX);
    }

    std::cout << std::endl;
    PRINT_BENCHMARK("strf::wwrite(dest) [{LLONG_MAX}]")
    {
        strf::wwrite(dest) [{LLONG_MAX}];
    }
    PRINT_BENCHMARK("fwprintf(dest, L\"%lld\", LLONG_MAX)")
    {
        fwprintf(dest, L"%lld", LLONG_MAX);
    }

    std::cout << std::endl;
    PRINT_BENCHMARK("strf::wwrite(dest) [{25, 25, 25}]")
    {
        strf::wwrite(dest) [{25, 25, 25}];
    }
    PRINT_BENCHMARK("strf::wwrite(dest) (L\"{}{}{}\") = {25, 25, 25}")
    {
        strf::wwrite(dest) (L"{}{}{}") = {25, 25, 25};
    }
    PRINT_BENCHMARK("fwprintf(dest, L\"%d%d%d\", 25, 25, 25)")
    {
        fwprintf(dest, L"%d%d%d", 25, 25, 25);
    }


    std::cout << std::endl;
    PRINT_BENCHMARK("strf::wwrite(dest) [{LLONG_MAX, LLONG_MAX, LLONG_MAX}]")
    {
        strf::wwrite(dest) [{LLONG_MAX, LLONG_MAX, LLONG_MAX}];
    }
    PRINT_BENCHMARK("fwprintf(dest, L\"%d%d%d\", LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        fwprintf(dest, L"%lld%lld%lld", LLONG_MAX, LLONG_MAX, LLONG_MAX);
    }


    std::cout << std::endl;
    PRINT_BENCHMARK("strf::wwrite(dest) [{{25, 20}}]")
    {
        strf::wwrite(dest)[{{25, 20}}];
    }
    PRINT_BENCHMARK("strf::wwrite(dest) .facets(strf::width(20)) [{25}]")
    {
        strf::wwrite(dest).facets(strf::width(20)) [{25}];
    }
    PRINT_BENCHMARK("strf::wwrite(dest) [{ {join_right(20), {25}} }]")
    {
        strf::wwrite(dest) [{ {strf::join_right(20), {25}} }];
    }
    PRINT_BENCHMARK("fwprintf(dest, L\"%20d\", 25)")
    {
        fwprintf(dest, L"%20d", 25);
    }


    std::cout << std::endl;
    PRINT_BENCHMARK("strf::wwrite(dest) [{{25, {6, L\"<+\"}}}]")
    {
        strf::wwrite(dest)[{{25, {6, "<+"}}}];
    }
    PRINT_BENCHMARK("strf::wwrite(dest).facets(width(6), left, showpos) [{ 25 }]")
    {
        strf::wwrite(dest).facets(strf::width(6), strf::left, strf::showpos) [{ 25 }];
    }

    PRINT_BENCHMARK("strf::wwrite(dest)({strf::make_ftuple(width(6), left, showpos), {25}})")
    {
        strf::wwrite(dest) [{ {strf::make_ftuple(strf::width(6), strf::left, strf::showpos), {25}} }];
    }
    PRINT_BENCHMARK("fwprintf(dest, L\"%6-+d\", 25)")
    {
        fwprintf(dest, L"%-+6d", 25);
    }


    std::cout << std::endl;
    PRINT_BENCHMARK("strf::wwrite(dest) [{{25, L\"#x\"}}]")
    {
        strf::wwrite(dest) [{{25, "#x"}}];
    }
    PRINT_BENCHMARK("strf::wwrite(dest) .facets(hex, showbase) [{25}]")
    {
        strf::wwrite(dest).facets(strf::hex, strf::showbase) [{25}];
    }
    PRINT_BENCHMARK("fwprintf(dest, L\"%#x\", 25)")
    {
        fwprintf(dest, L"%#x", 25);
    }


    std::cout << std::endl;
    PRINT_BENCHMARK("strf::wwrite(dest) [{25, +strf::fmt(25)<6, ~strf::hex(25)]")
    {
        strf::wwrite(dest) &= {25, +strf::fmt(25)<6, ~strf::hex(25)};
    }
    PRINT_BENCHMARK("fwprintf(dest, L\"%d%-+6d%#x\", 25, 25, 25)")
    {
        fwprintf(dest, L"%d%-+6d%#x", 25, 25, 25);
    }

    std::cout << "\n Strings and itegers mixed: \n";

    std::cout << std::endl;
    PRINT_BENCHMARK("strf::wwrite(dest) [{L\"ten =  \", 10, L\", twenty = \", 20}]")
    {
        strf::wwrite(dest) [{L"ten =  ", 10, L", twenty = ", 20}];
    }
    PRINT_BENCHMARK("strf::wwrite(dest) [{L\"ten =  \", 10, L\", twenty = \", 20}]")
    {
        strf::wwrite(dest) (L"ten = {}, twenty = {}") = {10, 20};
    }
    PRINT_BENCHMARK("fwprintf(dest, L\"ten = %d, twenty= %d\", 10, 20)")
    {
        fwprintf(dest, L"ten = %d, twenty= %d", 10, 20);
    }

    fclose(dest);
    return 1;
}
