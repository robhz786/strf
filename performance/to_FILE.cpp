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

    PRINT_BENCHMARK("strf::format(dest) .error_code(\"Hello \", \"World\", \"!\")")
    {
        (void)strf::format(dest) .error_code("Hello ", "World", "!");
    }
    PRINT_BENCHMARK("strf::format(dest) (\"Hello {}!\") .error_code(\"World\")")
    {
        (void)strf::format(dest) ("Hello {}!") .error_code("World");
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

        PRINT_BENCHMARK("strf::format(dest) .error_code(\"Hello \", long_string, \"!\")")
        {
            (void)strf::format(dest) .error_code("Hello ", long_string, "!");
        }
        PRINT_BENCHMARK("strf::format(dest) (\"Hello {}!\") .error_code(long_string)")
        {
            (void)strf::format(dest) ("Hello {}!") .error_code(long_string);
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

    PRINT_BENCHMARK("strf::format(dest) .error_code(strf::right(\"aa\", 20))")
    {
        (void)strf::format(dest) .error_code(strf::right("aa", 20));
    }
    PRINT_BENCHMARK("strf::format(dest) .error_code(join_right(20)(\"aa\"))")
    {
        (void)strf::format(dest) .error_code(strf::join_right(20)("aa"));
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

    PRINT_BENCHMARK("strf::format(dest) .error_code(25)")
    {
        (void)strf::format(dest) .error_code(25);
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
    PRINT_BENCHMARK("strf::format(dest) .error_code(LLONG_MAX)")
    {
        (void)strf::format(dest) .error_code(LLONG_MAX);
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
    PRINT_BENCHMARK("strf::format(dest).facets(numpunct_3) .error_code(LLONG_MAX)")
    {
        (void)strf::format(dest).facets(numpunct_3) .error_code(LLONG_MAX);
    }

    std::cout << std::endl;
    PRINT_BENCHMARK("strf::format(dest) .error_code(LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        (void)strf::format(dest) .error_code(LLONG_MAX, LLONG_MAX, LLONG_MAX);
    }
    PRINT_BENCHMARK("strf::format(dest) (\"{}{}{}\") .error_code(LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        (void)strf::format(dest) ("{}{}{}") .error_code(LLONG_MAX, LLONG_MAX, LLONG_MAX);
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

    PRINT_BENCHMARK("strf::format(dest) (\"{}{}{}\") .error_code(55555, +strf::fmt(55555)<8 , +strf::hex(55555))")
    {
        (void)strf::format(dest) ("{}{}{}") .error_code(55555, +strf::fmt(55555)<8 , +strf::hex(55555));
    }
    PRINT_BENCHMARK("strf::format(dest) .error_code(55555, +strf::fmt(55555)<8 , +strf::hex(55555))")
    {
        (void)strf::format(dest) .error_code(55555, +strf::fmt(55555)<8 , +strf::hex(55555));
    }
    PRINT_BENCHMARK("fmt::print(dest, \"{}{:<8}{:#x}\", 55555, 55555, 55555)")
    {
        fmt::print(dest, "{}{:<8}{:#x}", 55555, 55555, 55555);
    }
    PRINT_BENCHMARK("sprintf(dest, \"%d%-+8d%#x\", 55555, 55555, 55555)")
    {
        fprintf(dest, "%d%-+8d%#x", 55555, 55555, 55555);
    }


    std::cout << "\n Strings and itegers mixed: \n";

    PRINT_BENCHMARK("strf::format(dest) (\"blah blah {} blah {} blah {}\") .error_code(INT_MAX, ~strf::hex(1234)<8, \"abcdef\")")
    {
        (void)strf::format(dest) ("blah blah {} blah {} blah {}") .error_code(INT_MAX, ~strf::hex(1234)<8, "abcdef");
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

    PRINT_BENCHMARK("strf::format(dest) (\"ten = {}, twenty = {}\") .error_code(10, 20)")
    {
        (void)strf::format(dest) ("ten = {}, twenty = {}") .error_code(10, 20);
    }
    PRINT_BENCHMARK("strf::format(dest) .error_code(\"ten =  \", 10, \", twenty = \", 20)")
    {
        (void)strf::format(dest) .error_code("ten =  ", 10, ", twenty = ", 20);
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
    
        PRINT_BENCHMARK("strf::format(buff) .exception(u16sample1); strf::format(dest) .exception(buff)")
        {
            strf::format(buff) .exception(u16sample1);
            strf::format(dest) .exception(buff);
        }
        PRINT_BENCHMARK("strf::format(dest) .exception(u16sample1)")
        {
            strf::format(dest) .exception(u16sample1);
        }
        std::cout << "\n";
        PRINT_BENCHMARK("strf::format(buff) .exception(u16sample2); strf::format(dest) .exception(buff)")
        {
            strf::format(buff) .exception(u16sample2);
            strf::format(dest) .exception(buff);
        }
        PRINT_BENCHMARK("strf::format(dest) .exception(u16sample2)")
        {
            strf::format(dest) .exception(u16sample2);
        }
        std::cout << "\n";
        PRINT_BENCHMARK("strf::format(buff) .exception(u16sample3); strf::format(dest) .exception(buff)")
        {
            strf::format(buff) .exception(u16sample3);
            strf::format(dest) .exception(buff);
        }
        PRINT_BENCHMARK("strf::format(dest) .exception(u16sample3)")
        {
            strf::format(dest) .exception(u16sample3);
        }
    }
    
    fclose(dest);
    return 1;
}
