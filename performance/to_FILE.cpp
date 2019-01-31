//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <climits>
#include <stdio.h>
#include <boost/stringify.hpp>
#include "loop_timer.hpp"
#include "fmt/format.h"

int main()
{
    namespace strf = boost::stringify;

#ifdef _WIN32
    FILE* dest = fopen("NUL", "w");
#else
    FILE* dest = fopen("/dev/null", "w");
#endif

    std::cout << "\n small strings \n";

    PRINT_BENCHMARK("strf::write(dest) (\"Hello \", \"World\", \"!\")")
    {
        (void)strf::write(dest)("Hello ", "World", "!");
    }
    PRINT_BENCHMARK("strf::write(dest) .as(\"Hello {}!\", \"World\")")
    {
        (void)strf::write(dest) .as("Hello {}!", "World");
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

        PRINT_BENCHMARK("strf::write(dest) (\"Hello \", long_string, \"!\")")
        {
            (void)strf::write(dest)("Hello ", long_string, "!");
        }
        PRINT_BENCHMARK("strf::write(dest) .as(\"Hello {}!\", long_string)")
        {
            (void)strf::write(dest) .as("Hello {}!", long_string);
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

    std::cout << "\n padding \n";

    PRINT_BENCHMARK("strf::write(dest) (strf::right(\"aa\", 20))")
    {
        (void)strf::write(dest)(strf::right("aa", 20));
    }
    PRINT_BENCHMARK("strf::write(dest) (join_right(20)(\"aa\"))")
    {
        (void)strf::write(dest)(strf::join_right(20)("aa"));
    }
    PRINT_BENCHMARK("fmt::print(dest, \"{:20}\", \"aa\")")
    {
        fmt::print(dest, "{:20}", "aa");
    }
    PRINT_BENCHMARK("fprintf(dest, \"%20s\", \"aa\")")
    {
        fprintf(dest, "%20s", "aa");
    }

    std::cout << "\n integers \n";

    PRINT_BENCHMARK("strf::write(dest) (25)")
    {
        (void)strf::write(dest)(25);
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
    PRINT_BENCHMARK("strf::write(dest) (LLONG_MAX)")
    {
        (void)strf::write(dest) (LLONG_MAX);
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
    strf::monotonic_grouping<10> numpunct_3(3);
    PRINT_BENCHMARK("strf::write(dest) .facets(numpunct_3) (LLONG_MAX)")
    {
        (void)strf::write(dest) .facets(numpunct_3) (LLONG_MAX);
    }

    std::cout << std::endl;
    PRINT_BENCHMARK("strf::write(dest) (LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        (void)strf::write(dest)(LLONG_MAX, LLONG_MAX, LLONG_MAX);
    }
    PRINT_BENCHMARK("strf::write(dest) .as(\"{}{}{}\", LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        (void)strf::write(dest) .as("{}{}{}", LLONG_MAX, LLONG_MAX, LLONG_MAX);
    }
    PRINT_BENCHMARK("fmt::print(dest, \"{}{}{}\", LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        fmt::print(dest, "{}{}{}", LLONG_MAX, LLONG_MAX, LLONG_MAX);
    }
    PRINT_BENCHMARK("fprintf(dest, \"%d%d%d\", LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        fprintf(dest, "%lld%lld%lld", LLONG_MAX, LLONG_MAX, LLONG_MAX);
    }

    std::cout << "\n formatted integers \n";

    PRINT_BENCHMARK("strf::write(dest) .as(\"{}{}{}\", 55555, +strf::fmt(55555)<8 , +strf::hex(55555))")
    {
        (void)strf::write(dest) .as("{}{}{}", 55555, +strf::fmt(55555)<8 , +strf::hex(55555));
    }
    PRINT_BENCHMARK("strf::write(dest) (55555, +strf::fmt(55555)<8 , +strf::hex(55555))")
    {
        (void)strf::write(dest) (55555, +strf::fmt(55555)<8 , +strf::hex(55555));
    }
    PRINT_BENCHMARK("fmt::print(dest, \"{}{:<8}{:#x}\", 55555, 55555, 55555)")
    {
        fmt::print(dest, "{}{:<8}{:#x}", 55555, 55555, 55555);
    }
    PRINT_BENCHMARK("fprintf(dest, \"%d%-+8d%#x\", 55555, 55555, 55555)")
    {
        fprintf(dest, "%d%-+8d%#x", 55555, 55555, 55555);
    }


    std::cout << "\n Strings and integers mixed: \n";

    PRINT_BENCHMARK("strf::write(dest) .as(\"blah blah {} blah {} blah {}\", INT_MAX, ~strf::hex(1234)<8, \"abcdef\")")
    {
        (void)strf::write(dest) .as("blah blah {} blah {} blah {}", INT_MAX, ~strf::hex(1234)<8, "abcdef");
    }
    PRINT_BENCHMARK("fmt::print(dest, \"blah blah {} blah {:<#8x} blah {}\", INT_MAX, 1234, \"abcdef\")")
    {
        fmt::print(dest, "blah blah {} blah {:<#8x} blah {}", INT_MAX, 1234, "abcdef");
    }
    PRINT_BENCHMARK("fprintf(dest, \"blah blah %d blah %#-8x blah %s\", INT_MAX, 1234, \"abcdef\")")
    {
        fprintf(dest, "blah blah %d blah %#-8x blah %s", INT_MAX, 1234, "abcdef");
    }

    std::cout << std::endl;

    PRINT_BENCHMARK("strf::write(dest) .as(\"ten = {}, twenty = {}\", 10, 20)")
    {
        (void)strf::write(dest) .as("ten = {}, twenty = {}", 10, 20);
    }
    PRINT_BENCHMARK("strf::write(dest) (\"ten =  \", 10, \", twenty = \", 20)")
    {
        (void)strf::write(dest) ("ten =  ", 10, ", twenty = ", 20);
    }
    PRINT_BENCHMARK("fmt::print(dest, \"ten = {}, twenty = {}\", 10, 20)")
    {
        fmt::print(dest, "ten = {}, twenty = {}", 10, 20);
    }
    PRINT_BENCHMARK("fprintf(dest, \"ten = %d, twenty= %d\", 10, 20)")
    {
        fprintf(dest, "ten = %d, twenty= %d", 10, 20);
    }

    std::cout << "\n Converting UTF-16 to UTF8\n";
    {
        std::u16string u16sample1(500, u'A');
        std::u16string u16sample2(500, u'\u0100');
        std::u16string u16sample3(500, u'\u0800');
        char buff[100000];

        PRINT_BENCHMARK("strf::write(buff) (strf::cv(u16sample1)); strf::write(dest) (buff)")
        {
            (void)strf::write(buff) (strf::cv(u16sample1));
            (void)strf::write(dest) (buff);
        }
        PRINT_BENCHMARK("strf::write(dest) (strf::cv(u16sample1))")
        {
            (void)strf::write(dest) (strf::cv(u16sample1));
        }
        std::cout << "\n";
        PRINT_BENCHMARK("strf::write(buff) (strf::cv(u16sample2)); strf::write(dest) (buff)")
        {
            (void)strf::write(buff) (strf::cv(u16sample2));
            (void)strf::write(dest) (buff);
        }
        PRINT_BENCHMARK("strf::write(dest) (strf::cv(u16sample2)")
        {
            (void)strf::write(dest) (strf::cv(u16sample2));
        }
        std::cout << "\n";
        PRINT_BENCHMARK("strf::write(buff) (strf::cv(u16sample3); strf::write(dest) (buff)")
        {
            (void)strf::write(buff) (strf::cv(u16sample3));
            (void)strf::write(dest) (buff);
        }
        PRINT_BENCHMARK("strf::write(dest) (strf::cv(u16sample3)")
        {
            (void)strf::write(dest) (strf::cv(u16sample3));
        }
    }

    fclose(dest);
    return 1;
}
