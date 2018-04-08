//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <sstream>
#include <boost/stringify.hpp>
#include "loop_timer.hpp"
#include "fmt/format.h"
#include <climits>

int main()
{
    namespace strf = boost::stringify;

    PRINT_BENCHMARK("strf::make_string.no_reserve().error_code(\"Hello \", \"World\", \"!\")")
    {
        auto s = strf::make_string.no_reserve().error_code("Hello ", "World", "!");
        (void)s;
    }
    PRINT_BENCHMARK("strf::make_string.no_reserve().exception(\"Hello \", \"World\", \"!\")")
    {
        auto s = strf::make_string.no_reserve().exception("Hello ", "World", "!");
        (void)s;
    }
    PRINT_BENCHMARK("fmt::format(\"Hello {}!\", \"World\")")
    {
        auto s = fmt::format("Hello {}!", "World");
        (void)s;
    }

    std::cout << "\n";
    
    PRINT_BENCHMARK("strf::make_string.exception(25)")
    {
        auto s = strf::make_string.exception(25);
        (void)s;
    }
    PRINT_BENCHMARK("strf::make_string.reserve(2).exception(25)")
    {
        auto s = strf::make_string.reserve(2).exception(25);
        (void)s;
    }
    PRINT_BENCHMARK("strf::make_string.no_reserve().exception(25)")
    {
        std::string s = strf::make_string.no_reserve().exception(25);
        (void)s;
    }
    PRINT_BENCHMARK("fmt::format(\"{}\", 25)")
    {
        std::string s = fmt::format("{}", 25);
        (void)s;
    }
    PRINT_BENCHMARK("std::to_string(25)")
    {
        std::string s = std::to_string(25);
        (void)s;
    }

    std::cout << "\n";

    PRINT_BENCHMARK("strf::make_string.exception(LLONG_MAX)")
    {
        auto s = strf::make_string.exception(LLONG_MAX);
        (void)s;
    }
    PRINT_BENCHMARK("strf::make_string.no_reserve().exception(LLONG_MAX)")
    {
        auto s = strf::make_string.no_reserve().exception(LLONG_MAX);
        (void)s;
    }
    PRINT_BENCHMARK("strf::make_string.reserve(100).exception(LLONG_MAX)")
    {
        auto s = strf::make_string.reserve(100).exception(LLONG_MAX);
        (void)s;
    }
    PRINT_BENCHMARK("fmt::format(\"{}\", LLONG_MAX)")
    {
        auto s = fmt::format("{}", LLONG_MAX);
        (void)s;
    }
    PRINT_BENCHMARK("std::to_string(LLONG_MAX)")
    {
        auto s = std::to_string(LLONG_MAX);
        (void)s;
    }

    std::cout << "\n";

    PRINT_BENCHMARK("strf::make_string.exception(strf::right(\"aa\", 20))")
    {
        (void)strf::make_string.exception(strf::right("aa", 20));
    }
    PRINT_BENCHMARK("strf::make_string.reseve(20).exception(strf::right(\"aa\", 20))")
    {
        (void)strf::make_string.reserve(20).exception(strf::right("aa", 20));
    }

    PRINT_BENCHMARK("fmt::format(dest, \"{:20}\", \"aa\")")
    {
        fmt::format("{:20}", "aa");
    }
    
    std::cout << "\n";

    PRINT_BENCHMARK("strf::make_string.no_reserve().exception(\"ten = \", 10, \"twenty = \", 20)")
    {
        auto s = strf::make_string.no_reserve().exception("ten = ", 10, "twenty = ", 20);
        (void)s;
    }
    
    PRINT_BENCHMARK("strf::make_string (\"ten = {}, twenty = {}\").error_code(10, 20)")
    {
        auto s = strf::make_string ("ten = {}, twenty = {}").error_code(10, 20);
        (void)s;
    }
    PRINT_BENCHMARK("strf::make_string (\"ten = {}, twenty = {}\").exception(10, 20)")
    {
        auto s = strf::make_string ("ten = {}, twenty = {}").exception(10, 20);
        (void)s;
    }
    PRINT_BENCHMARK("strf::make_string .reserve(30) (\"ten = {}, twenty = {}\").exception(10, 20)")
    {
        auto s = strf::make_string ("ten = {}, twenty = {}").exception(10, 20);
        (void)s;
    }
    PRINT_BENCHMARK("fmt::format(\"ten = {}, twenty = {}\", 10, 20)")
    {
        auto s = fmt::format("ten = {}, twenty = {}", 10, 20);
        (void)s;
    }

    PRINT_BENCHMARK("oss << \"ten = \" << 10 << \", twenty = \" << 20")
    {
        std::ostringstream oss;
        oss << "ten = " << 10 << ", twenty = " << 20;
        (void)oss;
    }
    PRINT_BENCHMARK("oss << \"ten = \" << 10 << \", twenty = \" << 20 ; auto s = oss.str()")
    {
        std::ostringstream oss;
        oss << "ten = " << 10 << ", twenty = " << 20;
        std::string s = oss.str();
        (void)oss;
        (void)s;
    }

    {
        char buff[200];
        PRINT_BENCHMARK("sprintf(buff, \"ten = %d, twenty = %d\", 10, 20); std::string{buff}")
        {
            sprintf(buff, "ten = %d, twenty = %d", 10, 20);
            std::string s{buff};
            (void)s;
        }
    }
    std::cout << "\n Converting UTF-8 to UTF16\n";
    {
        char16_t buff[1024];
        std::string u8sample1(500, 'A');
        // std::string u8sample2;
        // std::string u8sample3;
        // std::string u8sample4;
        // for(int i = 0; i < 500; ++i) u8sample2.append(u8"\u0100");
        // for(int i = 0; i < 500; ++i) u8sample3.append(u8"\u0800");
        // for(int i = 0; i < 500; ++i) u8sample4.append(u8"\U00010000");

        PRINT_BENCHMARK("strf::make_u16string.no_reserve() .exception(u8sample1)")
        {
            strf::make_u16string.no_reserve().exception(u8sample1);
        }
        PRINT_BENCHMARK("strf::format(buff) .exception(u8sample1); strf::make_u16string.exception(buff)")
        {
            strf::format(buff).exception(u8sample1);
            strf::make_u16string.exception(buff);
        }
        PRINT_BENCHMARK("strf::make_u16string .exception(u8sample1)")
        {
            strf::make_u16string.exception(u8sample1);
        }
        PRINT_BENCHMARK("strf::make_u16string.reserve(510) .exception(u8sample1)")
        {
            strf::make_u16string.reserve(510).exception(u8sample1);
        }

    }

    std::cout << "\n Converting UTF-16 to UTF8\n";
    {
        char buff[2000];
        std::u16string u16sample1(500, u'A');
        // std::u16string u16sample2(500, u'\u0100');
        // std::u16string u16sample3(500, u'\u0800');

        PRINT_BENCHMARK("strf::make_string.no_reserve() .exception(u16sample1)")
        {
            strf::make_string.no_reserve().exception(u16sample1);
        }
        PRINT_BENCHMARK("strf::format(buff) .exception(u16sample1); strf::make_string.exception(buff)")
        {
            strf::format(buff).exception(u16sample1);
            strf::make_string.exception(buff);
        }
        PRINT_BENCHMARK("strf::make_string .exception(u16sample1)")
        {
            strf::make_string.exception(u16sample1);
        }
        PRINT_BENCHMARK("strf::make_string.reserve(510) .exception(u16sample1)")
        {
            strf::make_string.reserve(510).exception(u16sample1);
        }

    }
    
    return 0;
}
