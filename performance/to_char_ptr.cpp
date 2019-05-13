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
#define HAS_CHARCONV
#include <charconv>
#endif
#endif

int main()
{
    namespace strf = boost::stringify;
    char dest[1000000];
    char* dest_end = dest + sizeof(dest);
    (void) dest_end;

    std::cout << "\n small strings \n";
    PRINT_BENCHMARK("strf::ec_write(dest) (\"Hello World!\")")
    {
        (void)strf::ec_write(dest)("Hello World!");
    }
    PRINT_BENCHMARK("strf::write(dest) (\"Hello World!\")")
    {
        (void)strf::write(dest)("Hello World!");
    }
    PRINT_BENCHMARK("strf::write(dest) .tr(\"{}\", \"Hello World!\")")
    {
        (void)strf::write(dest) .tr("{}", "Hello World!");
    }
    PRINT_BENCHMARK("fmt::format_to(dest, \"{}\", \"Hello World!\")")
    {
        fmt::format_to(dest, "{}", "Hello World!");
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
    }

    std::cout << "\n";
    PRINT_BENCHMARK("strf::ec_write(dest) (\"Hello \", \"World\", '!')")
    {
        (void)strf::ec_write(dest)("Hello ", "World", '!');
    }
    PRINT_BENCHMARK("strf::write(dest) (\"Hello \", \"World\", '!')")
    {
        (void)strf::write(dest)("Hello ", "World", '!');
    }
    PRINT_BENCHMARK("strf::write(dest) (\"Hello \", \"World\", \"!\")")
    {
        (void)strf::write(dest)("Hello ", "World", "!");
    }
    PRINT_BENCHMARK("strf::write(dest) .tr(\"Hello {}!\", \"World\")")
    {
        (void)strf::write(dest) .tr("Hello {}!", "World");
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
        PRINT_BENCHMARK("strf::write(dest) .tr(\"Hello {}!\", long_string)")
        {
            (void)strf::write(dest) .tr("Hello {}!", long_string);
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

    PRINT_BENCHMARK_N(10, "strf::ec_write(dest) (25)")
    {
        (void)strf::ec_write(dest)(20);
        (void)strf::ec_write(dest)(21);
        (void)strf::ec_write(dest)(22);
        (void)strf::ec_write(dest)(23);
        (void)strf::ec_write(dest)(24);
        (void)strf::ec_write(dest)(25);
        (void)strf::ec_write(dest)(26);
        (void)strf::ec_write(dest)(27);
        (void)strf::ec_write(dest)(28);
        (void)strf::ec_write(dest)(29);
    }
    PRINT_BENCHMARK_N(10, "strf::write(dest) (25)")
    {
        (void)strf::write(dest)(20);
        (void)strf::write(dest)(21);
        (void)strf::write(dest)(22);
        (void)strf::write(dest)(23);
        (void)strf::write(dest)(24);
        (void)strf::write(dest)(25);
        (void)strf::write(dest)(26);
        (void)strf::write(dest)(27);
        (void)strf::write(dest)(28);
        (void)strf::write(dest)(29);
    }
    PRINT_BENCHMARK("strf::write(dest) (strf::fmt(25))")
    {
        (void)strf::write(dest)(strf::fmt(25));
    }
    PRINT_BENCHMARK("fmt::format_to(dest, \"{}\", 25)")
    {
        fmt::format_to(dest, "{}", 25);
    }
    PRINT_BENCHMARK_N(10, "strcpy(dest, fmt::format_int{25}.c_str())")
    {
        strcpy(dest, fmt::format_int{20}.c_str());
        strcpy(dest + 2, fmt::format_int{21}.c_str());
        strcpy(dest + 4, fmt::format_int{22}.c_str());
        strcpy(dest + 6, fmt::format_int{23}.c_str());
        strcpy(dest + 8, fmt::format_int{24}.c_str());
        strcpy(dest + 10, fmt::format_int{25}.c_str());
        strcpy(dest + 12, fmt::format_int{26}.c_str());
        strcpy(dest + 14, fmt::format_int{27}.c_str());
        strcpy(dest + 16, fmt::format_int{28}.c_str());
        strcpy(dest + 18, fmt::format_int{29}.c_str());
    }
    PRINT_BENCHMARK_N(10, "fmt::format_int{25}.c_str()")
    {
        fmt::format_int{20}.c_str();
        fmt::format_int{21}.c_str();
        fmt::format_int{22}.c_str();
        fmt::format_int{23}.c_str();
        fmt::format_int{24}.c_str();
        fmt::format_int{25}.c_str();
        fmt::format_int{26}.c_str();
        fmt::format_int{27}.c_str();
        fmt::format_int{28}.c_str();
        fmt::format_int{29}.c_str();
    }

#if defined(HAS_CHARCONV)

    PRINT_BENCHMARK_N(10, "std::to_chars(dest, dest_end, 25)")
    {
        std::to_chars(dest, dest_end, 20);
        std::to_chars(dest, dest_end, 21);
        std::to_chars(dest, dest_end, 21);
        std::to_chars(dest, dest_end, 23);
        std::to_chars(dest, dest_end, 24);
        std::to_chars(dest, dest_end, 25);
        std::to_chars(dest, dest_end, 26);
        std::to_chars(dest, dest_end, 27);
        std::to_chars(dest, dest_end, 28);
        std::to_chars(dest, dest_end, 29);
    }

#endif// ! defined(HAS_CHARCONV)

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
    PRINT_BENCHMARK_N(10, "strcpy(dest, fmt::format_int{LLONG_MAX}.c_str())")
    {
        strcpy(dest, fmt::format_int{LLONG_MAX}.c_str());
        strcpy(dest + 100, fmt::format_int{LLONG_MAX - 1}.c_str());
        strcpy(dest + 200, fmt::format_int{LLONG_MAX - 2}.c_str());
        strcpy(dest + 300, fmt::format_int{LLONG_MAX - 3}.c_str());
        strcpy(dest + 400, fmt::format_int{LLONG_MAX - 4}.c_str());
        strcpy(dest + 500, fmt::format_int{LLONG_MAX - 5}.c_str());
        strcpy(dest + 600, fmt::format_int{LLONG_MAX - 6}.c_str());
        strcpy(dest + 700, fmt::format_int{LLONG_MAX - 7}.c_str());
        strcpy(dest + 800, fmt::format_int{LLONG_MAX - 8}.c_str());
        strcpy(dest + 900, fmt::format_int{LLONG_MAX - 9}.c_str());

    }
    PRINT_BENCHMARK_N(10, "fmt::format_int{LLONG_MAX}.c_str()")
    {
        fmt::format_int{LLONG_MAX}.c_str();
        fmt::format_int{LLONG_MAX - 1}.c_str();
        fmt::format_int{LLONG_MAX - 2}.c_str();
        fmt::format_int{LLONG_MAX - 3}.c_str();
        fmt::format_int{LLONG_MAX - 4}.c_str();
        fmt::format_int{LLONG_MAX - 5}.c_str();
        fmt::format_int{LLONG_MAX - 6}.c_str();
        fmt::format_int{LLONG_MAX - 7}.c_str();
        fmt::format_int{LLONG_MAX - 8}.c_str();
        fmt::format_int{LLONG_MAX - 9}.c_str();
    }

#if defined(HAS_CHARCONV)

    PRINT_BENCHMARK_N(10, "std::to_chars(dest, dest_end, LONG_MAX)")
    {
        std::to_chars(dest, dest_end, LONG_MAX);
        std::to_chars(dest, dest_end, LONG_MAX - 1);
        std::to_chars(dest, dest_end, LONG_MAX - 2);
        std::to_chars(dest, dest_end, LONG_MAX - 3);
        std::to_chars(dest, dest_end, LONG_MAX - 4);
        std::to_chars(dest, dest_end, LONG_MAX - 5);
        std::to_chars(dest, dest_end, LONG_MAX - 6);
        std::to_chars(dest, dest_end, LONG_MAX - 7);
        std::to_chars(dest, dest_end, LONG_MAX - 8);
        std::to_chars(dest, dest_end, LONG_MAX - 9);
    }

#endif // defined(HAS_CHARCONV)

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

#if defined(__GNU_LIBRARY__)
    PRINT_BENCHMARK("std::sprintf(dest, \"%'lld\", LLONG_MAX)")
    {
        std::sprintf(dest, "%'lld", LLONG_MAX);
    }
#else
    std::cout << "\n";
#endif
    std::cout << "\n";

/*
    std::cout << std::endl;
    PRINT_BENCHMARK("strf::write(dest) (25, 25, 25)")
    {
        (void)strf::write(dest) (25, 25, 25);
    }
    PRINT_BENCHMARK("strf::write(dest) .tr(\"{}{}{}\", 25, 25, 25)")
    {
        (void)strf::write(dest) .tr("{}{}{}", 25, 25, 25);
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
//     std::cout << std::endl;
    PRINT_BENCHMARK("strf::write(dest) (LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        (void)strf::write(dest) (LLONG_MAX, LLONG_MAX, LLONG_MAX);
    }
    PRINT_BENCHMARK("strf::write(dest) .tr(\"{}{}{}\", LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        (void)strf::write(dest) .tr("{}{}{}", LLONG_MAX, LLONG_MAX, LLONG_MAX);
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
    PRINT_BENCHMARK("strf::write(dest) .tr(\"{}{}{}\", 55555, +strf::left(55555, 8) , ~strf::hex(55555))")
    {
        (void)strf::write(dest) .tr("{}{}{}", 55555, +strf::left(55555, 8) , ~strf::hex(55555));
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

    PRINT_BENCHMARK("strf::write(dest) .tr(\"blah blah {} blah {} blah {}\", INT_MAX, ~strf::hex(1234)<8, \"abcdef\")")
    {
        (void)strf::write(dest).tr("blah blah {} blah {} blah {}", INT_MAX, ~strf::hex(1234)<8, "abcdef");
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
    PRINT_BENCHMARK("strf::write(dest) .tr(\"ten = {}, twenty = {}\", 10, 20)")
    {
        (void)strf::write(dest).tr("ten = {}, twenty = {}", 10, 20);
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
