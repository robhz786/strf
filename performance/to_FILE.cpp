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


int main()
{
    namespace strf = boost::stringify;

    FILE* dest = fopen("/dev/null", "w");

    std::cout << "\n small strings \n";

    PRINT_BENCHMARK("write_to(dest) = {\"Hello \", \"World\", \"!\"}")
    {
        strf::write_to(dest) = {"Hello ", "World", "!"};
    }
    PRINT_BENCHMARK("write_to(dest) [\"Hello {}!\"] = {\"World\"}")
    {
        strf::write_to(dest) ["Hello {}!"] = {"World"};
    }
    PRINT_BENCHMARK("fmt::print(dest, \"Hello {}!\", \"World\")")
    {
        fmt::print(dest, "Hello {}!", "World");
    }
    PRINT_BENCHMARK("fprintf(dest, \"Hello %s!\", \"World\")")
    {
        fprintf(dest, "Hello %s!", "World");
    }

    std::cout << "\n long string ( 1000 characters ): \n";

    {
        std::string std_string_long_string(1000, 'x');
        const char* long_string = std_string_long_string.c_str();

        PRINT_BENCHMARK("write_to(dest) = {\"Hello \", long_string, \"!\"}")
        {
            strf::write_to(dest) = {"Hello ", long_string, "!"};
        }
        PRINT_BENCHMARK("fmt::print(dest, \"Hello {}!\", long_string)")
        {
            fmt::print(dest, "Hello {}!", long_string);
        }
        PRINT_BENCHMARK("fprintf(dest, \"Hello %s!\", long_string)")
        {
            fprintf(dest, "Hello %s!", long_string);
        }
    }

    std::cout << "\n integers \n";

    PRINT_BENCHMARK("write_to(dest) = {25}")
    {
        strf::write_to(dest) = {25};
    }
    PRINT_BENCHMARK("fmt::print(dest, \"{}\", 25)")
    {
        fmt::print(dest, "{}", 25);
    }
    PRINT_BENCHMARK("fprintf(dest, \"%d\", 25)")
    {
        fprintf(dest, "%d", 25);
    }

    std::cout << std::endl;
    PRINT_BENCHMARK("write_to(dest) = {INT_MAX}")
    {
        strf::write_to(dest) = {INT_MAX};
    }
    PRINT_BENCHMARK("fmt::print(dest, \"{}\", INT_MAX)")
    {
        fmt::print(dest, "{}", INT_MAX);
    }
    PRINT_BENCHMARK("fprintf(dest, \"%d\", INT_MAX)")
    {
        fprintf(dest, "%d", INT_MAX);
    }

    std::cout << std::endl;
    PRINT_BENCHMARK("write_to(dest) = {LLONG_MAX}")
    {
        strf::write_to(dest) = {LLONG_MAX};
    }
    PRINT_BENCHMARK("fmt::print(dest, \"{}\", LLONG_MAX)")
    {
        fmt::print(dest, "{}", LLONG_MAX);
    }
    PRINT_BENCHMARK("fprintf(dest, \"%lld\", LLONG_MAX)")
    {
        fprintf(dest, "%lld", LLONG_MAX);
    }

    std::cout << std::endl;
    PRINT_BENCHMARK("write_to(dest) = {25, 25, 25}")
    {
        strf::write_to(dest) = {25, 25, 25};
    }
    PRINT_BENCHMARK("write_to(dest) [\"{}{}{}\"] = {25, 25, 25}")
    {
        strf::write_to(dest) ["{}{}{}"] = {25, 25, 25};
    }

    PRINT_BENCHMARK("fmt::print(dest, \"{}{}{}\", 25, 25, 25)")
    {
        fmt::print(dest, "{}{}{}", 25, 25, 25);
    }
    PRINT_BENCHMARK("fprintf(dest, \"%d%d%d\", 25, 25, 25)")
    {
        fprintf(dest, "%d%d%d", 25, 25, 25);
    }


    std::cout << std::endl;
    PRINT_BENCHMARK("write_to(dest) = {LLONG_MAX, LLONG_MAX, LLONG_MAX}")
    {
        strf::write_to(dest) = {LLONG_MAX, LLONG_MAX, LLONG_MAX};
    }
    PRINT_BENCHMARK("fmt::print(dest, \"{}{}{}\", LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        fmt::print(dest, "{}{}{}", LLONG_MAX, LLONG_MAX, LLONG_MAX);
    }
    PRINT_BENCHMARK("fprintf(dest, \"%d%d%d\", LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        fprintf(dest, "%lld%lld%lld", LLONG_MAX, LLONG_MAX, LLONG_MAX);
    }


    std::cout << std::endl;
    PRINT_BENCHMARK("write_to(dest) = {{25, 20}}")
    {
        strf::write_to(dest)= {{25, 20}};
    }
    PRINT_BENCHMARK("write_to(dest) .with(strf::width(20)) = {25}")
    {
        strf::write_to(dest).with(strf::width(20)) = {25};
    }
    PRINT_BENCHMARK("write_to(dest) = { {join_right(20), {25}} }")
    {
        strf::write_to(dest) = { {strf::join_right(20), {25}} };
    }
    PRINT_BENCHMARK("fmt::print(dest, \"{:20}\", 25)")
    {
        fmt::print(dest, "{:20}", 25);
    }
    PRINT_BENCHMARK("fprintf(dest, \"%20d\", 25)")
    {
        fprintf(dest, "%20d", 25);
    }


    std::cout << std::endl;
    PRINT_BENCHMARK("write_to(dest) = {{25, {6, \"<+\"}}}")
    {
        strf::write_to(dest)= {{25, {6, "<+"}}};
    }
    PRINT_BENCHMARK("write_to(dest).with(width(6), left, showpos) = { 25 }")
    {
        strf::write_to(dest).with(strf::width(6), strf::left, strf::showpos) = { 25 };
    }

    PRINT_BENCHMARK("write_to(dest)({strf::make_ftuple(width(6), left, showpos), {25}})")
    {
        strf::write_to(dest) = { {strf::make_ftuple(strf::width(6), strf::left, strf::showpos), {25}} };
    }
    PRINT_BENCHMARK("fmt::print(dest, \"{:<+6}\", 25)")
    {
        fmt::print(dest, "{:<+6}", 25);
    }
    PRINT_BENCHMARK("fprintf(dest, \"%6-+d\", 25)")
    {
        fprintf(dest, "%-+6d", 25);
    }


    std::cout << std::endl;
    PRINT_BENCHMARK("write_to(dest) = {{25, \"#x\"}}")
    {
        strf::write_to(dest) = {{25, "#x"}};
    }
    PRINT_BENCHMARK("write_to(dest) .with(hex, showbase) = {25}")
    {
        strf::write_to(dest).with(strf::hex, strf::showbase) = {25};
    }
    PRINT_BENCHMARK("fmt::print(dest, \"{:#x}\", 25)")
    {
        fmt::print(dest, "{:#x}", 25);
    }
    PRINT_BENCHMARK("fprintf(dest, \"%#x\", 25)")
    {
        fprintf(dest, "%#x", 25);
    }


    std::cout << std::endl;
    PRINT_BENCHMARK("write_to(dest) = {25, {25, {6, \"<+\"}} , {25, \"#x\"}}")
    {
        strf::write_to(dest) = {25, {25, {6, "<+"}} , {25, "#x"}};
    }
    PRINT_BENCHMARK("fmt::print(dest, \"{}{:<6}{:#x}\", 25, 25, 25)")
    {
        fmt::print(dest, "{}{:<6}{:#x}", 25, 25, 25);
    }
    PRINT_BENCHMARK("fprintf(dest, \"%d%-+6d%#x\", 25, 25, 25)")
    {
        fprintf(dest, "%d%-+6d%#x", 25, 25, 25);
    }

    std::cout << "\n Strings and itegers mixed: \n";

    std::cout << std::endl;
    PRINT_BENCHMARK("write_to(dest) = {\"ten =  \", 10, \", twenty = \", 20}")
    {
        strf::write_to(dest) = {"ten =  ", 10, ", twenty = ", 20};
    }
    PRINT_BENCHMARK("fmt::print(dest, \"ten = {}, twenty = {}\", 10, 20)")
    {
        fmt::print(dest, "ten = {}, twenty = {}", 10, 20);
    }
    PRINT_BENCHMARK("fprintf(dest, \"ten = %d, twenty= %d\", 10, 20)")
    {
        fprintf(dest, "ten = %d, twenty= %d", 10, 20);
    }


    fclose(dest);
    return 1;
}
