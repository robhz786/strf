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

#if defined(__has_include)
#if __has_include(<charconv>)
#if (__cplusplus >= 201402L) || (defined(_HAS_CXX17) && _HAS_CXX17)
#define HAS_CHARCONV
#include <charconv>
#endif
#endif
#endif

int main()
{
    namespace strf = boost::stringify;
    char dest[1000000];
    char* dest_end = dest + sizeof(dest);
    (void) dest_end;
    escape(dest);

    std::cout << "\n small strings \n";
    PRINT_BENCHMARK("strf::write(dest) (\"Hello World!\")")
    {
        (void)strf::write(dest)("Hello World!");
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, \"{}\", \"Hello World!\")")
    {
        fmt::format_to(dest, "{}", "Hello World!");
        clobber();
    }
    PRINT_BENCHMARK_N(10, "std::strcpy(dest, \"Hello World!\")")
    {
        std::strcpy(dest, "Hello World!");
        std::strcpy(dest + 12, "Hallo world!");
        std::strcpy(dest + 24, "Hellooooooo!");
        std::strcpy(dest + 36, "Hello!!!!!!!");
        std::strcpy(dest + 48, "ASDFSADFrld!");
        std::strcpy(dest + 60, "Heasdfsdfsad");
        std::strcpy(dest + 72, "Helasdfasf!!");
        std::strcpy(dest + 84, "Hallo asdfg!");
        std::strcpy(dest + 96, "abcdefghijkl");
        std::strcpy(dest + 108, "012345678901");
        clobber();
    }

    std::cout << "\n";
    PRINT_BENCHMARK("strf::write(dest) (\"Hello \", \"World\", '!')")
    {
        (void)strf::write(dest)("Hello ", "World", '!');
        clobber();
    }
    PRINT_BENCHMARK("strf::write(dest) (\"Hello \", \"World\", \"!\")")
    {
        (void)strf::write(dest)("Hello ", "World", "!");
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, \"Hello {}!\", \"World\")")
    {
        fmt::format_to(dest, "Hello {}!", "World");
        clobber();
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"Hello %s!\", \"World\")")
    {
        std::sprintf(dest, "Hello %s!", "World");
        clobber();
    }

    std::cout << "\n long string ( 1000 characters ): \n";

    {
        std::string std_string_long_string(1000, 'x');
        const char* long_string = std_string_long_string.c_str();

        PRINT_BENCHMARK("strf::write(dest) (\"Hello \", long_string, \"!\")")
        {
            (void)strf::write(dest)("Hello ", long_string, "!");
            clobber();
        }
        PRINT_BENCHMARK("fmt::format_to(dest, \"Hello {}!\", long_string)")
        {
            fmt::format_to(dest, "Hello {}!", long_string);
            clobber();
        }
        PRINT_BENCHMARK("std::sprintf(dest, \"Hello %s!\", long_string)")
        {
            std::sprintf(dest, "Hello %s!", long_string);
            clobber();
        }
    }

    std::cout << "\n padding \n";

    PRINT_BENCHMARK("strf::write(dest) (strf::right(\"aa\", 20))")
    {
        (void)strf::write(dest)(strf::right("aa", 20));
        clobber();
    }
    PRINT_BENCHMARK("strf::write(dest) (join_right(20)(\"aa\"))")
    {
        (void)strf::write(dest)(strf::join_right(20)("aa"));
    }
    PRINT_BENCHMARK("fmt::format_to(dest, \"{:20}\", \"aa\")")
    {
        fmt::format_to(dest, "{:20}", "aa");
        clobber();
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"%20s\", \"aa\")")
    {
        std::sprintf(dest, "%20s", "aa");
        clobber();
    }


    std::cout << "\n Strings and integers mixed: \n";

    PRINT_BENCHMARK("strf::write(dest) (\"blah blah \", INT_MAX, \" blah \", ~strf::hex(1234)<8, \" blah \", \"abcdef\")" )
    {
        (void)strf::write(dest)("blah blah ", INT_MAX, " blah ", ~strf::hex(1234)<8, " blah ", "abcdef");
        clobber();
    }

    PRINT_BENCHMARK("strf::write(dest) .tr(\"blah blah {} blah {} blah {}\", INT_MAX, ~strf::hex(1234)<8, \"abcdef\")")
    {
        (void)strf::write(dest).tr("blah blah {} blah {} blah {}", INT_MAX, ~strf::hex(1234)<8, "abcdef");
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, \"blah blah {} blah {:<#8x} blah {}\", INT_MAX, 1234, \"abcdef\")")
    {
        fmt::format_to(dest, "blah blah {} blah {:<#8x} blah {}", INT_MAX, 1234, "abcdef");
        clobber();
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"blah blah %d blah %#-8x blah %s\", INT_MAX, 1234, \"abcdef\")")
    {
        std::sprintf(dest, "blah blah %d blah %#-8x blah %s", INT_MAX, 1234, "abcdef");
        clobber();
    }

    std::cout << std::endl;
    PRINT_BENCHMARK("strf::write(dest) (\"ten =  \", 10, \", twenty = \", 20)")
    {
        (void)strf::write(dest)("ten =  ", 10, ", twenty = ", 20);
        clobber();
    }
    PRINT_BENCHMARK("strf::write(dest) .tr(\"ten = {}, twenty = {}\", 10, 20)")
    {
        (void)strf::write(dest).tr("ten = {}, twenty = {}", 10, 20);
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, \"ten = {}, twenty = {}\", 10, 20)")
    {
        fmt::format_to(dest, "ten = {}, twenty = {}", 10, 20);
        clobber();
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"ten = %d, twenty= %d\", 10, 20)")
    {
        std::sprintf(dest, "ten = %d, twenty= %d", 10, 20);
        clobber();
    }


    return 1;
}
