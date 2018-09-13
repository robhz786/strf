//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <clocale>
#include <stdio.h>
#include <cstring>
#include <climits>
#include <boost/stringify.hpp>
#include "loop_timer.hpp"
#include "fmt/format.h"


int main()
{
    namespace strf = boost::stringify;
    char dest[1000000];
    constexpr std::size_t dest_size = sizeof(dest);
    char* dest_end = dest + dest_size;

    std::cout << "\n small strings \n";
    PRINT_BENCHMARK("strf::write(dest) (\"Hello World!\")")
    {
        (void)strf::write(dest)("Hello World!");
    }
    PRINT_BENCHMARK("strf::write(dest) .as(\"{}\") (\"Hello World!\")")
    {
        (void)strf::write(dest) .as("{}")("Hello World!");
    }
    PRINT_BENCHMARK("fmt::format_to(dest, \"{}\", \"Hello World!\")")
    {
        fmt::format_to(dest, "{}", "Hello World!");
    }
    PRINT_BENCHMARK("std::strcpy(dest, \"Hello World!\")")
    {
        std::strcpy(dest, "Hello World!");
    }

    std::cout << "\n";
    PRINT_BENCHMARK("strf::write(dest) (\"Hello \", \"World\", \"!\")")
    {
        (void)strf::write(dest)("Hello ", "World", "!");
    }
    PRINT_BENCHMARK("strf::write(dest) .as(\"Hello {}!\") (\"World\")")
    {
        (void)strf::write(dest) .as("Hello {}!")("World");
    }
    PRINT_BENCHMARK("fmt::format_to(dest, \"Hello {}!\", \"World\")")
    {
        fmt::format_to(dest, "Hello {}!", "World");
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"Hello %s!\", \"World\")")
    {
        std::sprintf(dest, "Hello %s!", "World");
    }

    std::cout << "\n long string ( 1000 characters ): \n";

    {
        std::string std_string_long_string(1000, 'x');
        const char* long_string = std_string_long_string.c_str();

        PRINT_BENCHMARK("strf::write(dest) (\"Hello \", long_string, \"!\")")
        {
            (void)strf::write(dest)("Hello ", long_string, "!");
        }
        PRINT_BENCHMARK("strf::write(dest) .as(\"Hello {}!\") (long_string)")
        {
            (void)strf::write(dest) .as("Hello {}!")(long_string);
        }
        PRINT_BENCHMARK("fmt::format_to(dest, \"Hello {}!\", long_string)")
        {
            fmt::format_to(dest, "Hello {}!", long_string);
        }
        PRINT_BENCHMARK("std::sprintf(dest, \"Hello %s!\", long_string)")
        {
            std::sprintf(dest, "Hello %s!", long_string);
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
    PRINT_BENCHMARK("fmt::format_to(dest, \"{:20}\", \"aa\")")
    {
        fmt::format_to(dest, "{:20}", "aa");
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"%20s\", \"aa\")")
    {
        std::sprintf(dest, "%20s", "aa");
    }

    std::cout << "\n integers \n";

    PRINT_BENCHMARK("strf::write(dest) (25)")
    {
        (void)strf::write(dest)(25);
    }
    PRINT_BENCHMARK("fmt::format_to(dest, \"{}\", 25)")
    {
        fmt::format_to(dest, "{}", 25);
    }

#ifdef BOOST_STRINGIFY_HAS_STD_CHARCONV

    PRINT_BENCHMARK("std::to_chars(dest, dest_end, 25)")
    {
        std::to_chars(dest, dest_end, 25);
    }

#endif

    PRINT_BENCHMARK("std::sprintf(dest, \"%d\", 25)")
    {
        std::sprintf(dest, "%d", 25);
    }

    std::cout << std::endl;
    PRINT_BENCHMARK("strf::write(dest) (LLONG_MAX)")
    {
        (void)strf::write(dest)(LLONG_MAX);
    }
    PRINT_BENCHMARK("fmt::format_to(dest, \"{}\", LLONG_MAX)")
    {
        fmt::format_to(dest, "{}", LLONG_MAX);
    }

#ifdef BOOST_STRINGIFY_HAS_STD_CHARCONV

    PRINT_BENCHMARK("std::to_chars(dest, dest_end, LONG_MAX)")
    {
        std::to_chars(dest, dest_end, LONG_MAX);
    }

#endif

    PRINT_BENCHMARK("std::sprintf(dest, \"%lld\", LLONG_MAX)")
    {
        std::sprintf(dest, "%lld", LLONG_MAX);
    }

    std::cout << std::endl;
    std::setlocale(LC_ALL, "en_US.UTF-8");
    strf::monotonic_grouping<10> numpunct_3(3);
    PRINT_BENCHMARK("strf::write(dest) .facets(numpunct_3) (LLONG_MAX)")
    {
        (void)strf::write(dest).facets(numpunct_3)(LLONG_MAX);
    }
    PRINT_BENCHMARK("fmt::format_to(dest, \"{:n}\", LLONG_MAX)")
    {
        fmt::format_to(dest, "{:n}", LLONG_MAX);
    }

#if defined(__GNU_LIBRARY)
    PRINT_BENCHMARK("std::sprintf(dest, \"%'lld\", LLONG_MAX)")
    {
        std::sprintf(dest, "%'lld", LLONG_MAX);
    }
#else
    std::cout << "\n";
#endif

    /*
    std::cout << std::endl;
    PRINT_BENCHMARK("strf::write(dest) (25, 25, 25)")
    {
        (void)strf::write(dest) (25, 25, 25);
    }
    PRINT_BENCHMARK("strf::write(dest) .as(\"{}{}{}\") (25, 25, 25)")
    {
        (void)strf::write(dest) .as("{}{}{}") (25, 25, 25);
    }
    PRINT_BENCHMARK("fmt::format_to(dest, \"{}{}{}\", 25, 25, 25)")
    {
        fmt::format_to(dest, "{}{}{}", 25, 25, 25);
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"%d%d%d\", 25, 25, 25)")
    {
        std::sprintf(dest, "%d%d%d", 25, 25, 25);
    }
*/

    std::cout << std::endl;
    PRINT_BENCHMARK("strf::write(dest) (LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        (void)strf::write(dest) (LLONG_MAX, LLONG_MAX, LLONG_MAX);
    }
    PRINT_BENCHMARK("strf::write(dest) .as(\"{}{}{}\") (LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        (void)strf::write(dest) .as("{}{}{}") (LLONG_MAX, LLONG_MAX, LLONG_MAX);
    }
    PRINT_BENCHMARK("fmt::format_to(dest, \"{}{}{}\", LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        fmt::format_to(dest, "{}{}{}", LLONG_MAX, LLONG_MAX, LLONG_MAX);
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"%d%d%d\", LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        std::sprintf(dest, "%lld%lld%lld", LLONG_MAX, LLONG_MAX, LLONG_MAX);
    }

    std::cout << "\n formatted integers \n";
    PRINT_BENCHMARK("strf::write(dest) (55555, +strf::left(55555, 8) , ~strf::hex(55555))")
    {
        (void)strf::write(dest) (55555, +strf::left(55555, 8) , ~strf::hex(55555));
    }
    PRINT_BENCHMARK("strf::write(dest) .as(\"{}{}{}\") (55555, +strf::left(55555, 8) , ~strf::hex(55555))")
    {
        (void)strf::write(dest) .as("{}{}{}") (55555, +strf::left(55555, 8) , ~strf::hex(55555));
    }
    PRINT_BENCHMARK("fmt::format_to(dest, \"{}{:<8}{:#x}\", 55555, 55555, 55555)")
    {
        fmt::format_to(dest, "{}{:<8}{:#x}", 55555, 55555, 55555);
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"%d%-+8d%#x\", 55555, 55555, 55555)")
    {
        std::sprintf(dest, "%d%-+8d%#x", 55555, 55555, 55555);
    }

    std::cout << "\n Strings and integers mixed: \n";

    PRINT_BENCHMARK("strf::write(dest) (\"blah blah \", INT_MAX, \" blah \", ~strf::hex(1234)<8, \" blah \", \"abcdef\")" )
    {
        (void)strf::write(dest)("blah blah ", INT_MAX, " blah ", ~strf::hex(1234)<8, " blah ", "abcdef");
    }

    PRINT_BENCHMARK("strf::write(dest) .as(\"blah blah {} blah {} blah {}\") (INT_MAX, ~strf::hex(1234)<8, \"abcdef\")")
    {
        (void)strf::write(dest).as("blah blah {} blah {} blah {}")(INT_MAX, ~strf::hex(1234)<8, "abcdef");
    }
    PRINT_BENCHMARK("fmt::format_to(dest, \"blah blah {} blah {:<#8x} blah {}\", INT_MAX, 1234, \"abcdef\")")
    {
        fmt::format_to(dest, "blah blah {} blah {:<#8x} blah {}", INT_MAX, 1234, "abcdef");
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"blah blah %d blah %#-8x blah %s\", INT_MAX, 1234, \"abcdef\")")
    {
        std::sprintf(dest, "blah blah %d blah %#-8x blah %s", INT_MAX, 1234, "abcdef");
    }

    std::cout << std::endl;
    PRINT_BENCHMARK("strf::write(dest) (\"ten =  \", 10, \", twenty = \", 20)")
    {
        (void)strf::write(dest)("ten =  ", 10, ", twenty = ", 20);
    }
    PRINT_BENCHMARK("strf::write(dest) .as(\"ten = {}, twenty = {}\") (10, 20)")
    {
        (void)strf::write(dest).as("ten = {}, twenty = {}")(10, 20);
    }
    PRINT_BENCHMARK("fmt::format_to(dest, \"ten = {}, twenty = {}\", 10, 20)")
    {
        fmt::format_to(dest, "ten = {}, twenty = {}", 10, 20);
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"ten = %d, twenty= %d\", 10, 20)")
    {
        std::sprintf(dest, "ten = %d, twenty= %d", 10, 20);
    }


    return 1;
}
