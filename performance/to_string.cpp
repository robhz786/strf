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

    PRINT_BENCHMARK("strf::to_string (\"Hello \", \"World\", \"!\")")
    {
        (void)strf::to_string("Hello ", "World", "!");
    }
    PRINT_BENCHMARK("fmt::format(\"Hello {}!\", \"World\")")
    {
        (void)fmt::format("Hello {}!", "World");
    }

    std::cout << "\n";

    PRINT_BENCHMARK("strf::to_string(25)")
    {
        (void)strf::to_string(25);
    }
    PRINT_BENCHMARK("fmt::format(\"{}\", 25)")
    {
        std::string s = fmt::format("{}", 25);
    }
    PRINT_BENCHMARK("std::to_string(25)")
    {
        std::string s = std::to_string(25);
    }

    std::cout << "\n";

    PRINT_BENCHMARK("strf::to_string (LLONG_MAX)")
    {
        (void)strf::to_string(LLONG_MAX);
    }
    PRINT_BENCHMARK("strf::to_string .reserve_calc() (LLONG_MAX)")
    {
        (void)strf::to_string.reserve_calc()(LLONG_MAX);
    }
    PRINT_BENCHMARK("strf::to_string .reserve(100) (LLONG_MAX)")
    {
        (void)strf::to_string.reserve(100)(LLONG_MAX);
    }
    PRINT_BENCHMARK("fmt::format(\"{}\", LLONG_MAX)")
    {
        (void)fmt::format("{}", LLONG_MAX);
    }
    PRINT_BENCHMARK("std::to_string(LLONG_MAX)")
    {
        (void)std::to_string(LLONG_MAX);
    }

    std::cout << "\n";

    PRINT_BENCHMARK("strf::to_string(strf::right(\"aa\", 20))")
    {
        (void)strf::to_string(strf::right("aa", 20));
    }
    PRINT_BENCHMARK("strf::to_string .reserve(20) (strf::right(\"aa\", 20))")
    {
        (void)strf::to_string.reserve(20)(strf::right("aa", 20));
    }

    PRINT_BENCHMARK("fmt::format(dest, \"{:20}\", \"aa\")")
    {
        fmt::format("{:20}", "aa");
    }

    std::cout << "\n";

    PRINT_BENCHMARK("strf::to_string (\"ten = \", 10, \"twenty = \", 20)")
    {
        (void)strf::to_string ("ten = ", 10, "twenty = ", 20);
    }
    PRINT_BENCHMARK("strf::to_string .tr(\"ten = {}, twenty = {}\", 10, 20)")
    {
        (void)strf::to_string.tr("ten = {}, twenty = {}", 10, 20);
    }
    PRINT_BENCHMARK("strf::to_string .reserve(30) .tr(\"ten = {}, twenty = {}\", 10, 20)")
    {
        (void)strf::to_string.tr("ten = {}, twenty = {}", 10, 20);
    }
    PRINT_BENCHMARK("strf::to_string .reserve_calc() .tr(\"ten = {}, twenty = {}\", 10, 20)")
    {
        (void)strf::to_string.reserve_calc().tr("ten = {}, twenty = {}", 10, 20);
    }

    PRINT_BENCHMARK("fmt::format(\"ten = {}, twenty = {}\", 10, 20)")
    {
        (void)fmt::format("ten = {}, twenty = {}", 10, 20);
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

        PRINT_BENCHMARK("strf::to_u16string (strf::cv(u8sample1))")
        {
            (void)strf::to_u16string(strf::cv(u8sample1));
        }
        PRINT_BENCHMARK("strf::write(buff) (strf::cv(u8sample1)); strf::to_u16string.reserve_calc() (buff)")
        {
            (void)strf::write(buff)(strf::cv(u8sample1));
            (void)strf::to_u16string.reserve_calc() (buff);
        }
        PRINT_BENCHMARK("strf::to_u16string.reserve_calc() (strf::cv(u8sample1))")
        {
            (void)strf::to_u16string.reserve_calc()(strf::cv(u8sample1));
        }
        PRINT_BENCHMARK("strf::to_u16string .reserve(510) (strf::cv(u8sample1))")
        {
            (void)strf::to_u16string.reserve(510)(strf::cv(u8sample1));
        }

    }

    std::cout << "\n Converting UTF-16 to UTF8\n";
    {
        char buff[2000];
        std::u16string u16sample1(500, u'A');
        // std::u16string u16sample2(500, u'\u0100');
        // std::u16string u16sample3(500, u'\u0800');

        PRINT_BENCHMARK("strf::to_string (strf::cv(u16sample1))")
        {
            (void)strf::to_string(strf::cv(u16sample1));
        }
        PRINT_BENCHMARK("strf::write(buff) (strf::cv(u16sample1)); strf::to_string.reserve_calc()(buff)")
        {
            (void)strf::write(buff)(strf::cv(u16sample1));
            (void)strf::to_string.reserve_calc()(buff);
        }
        PRINT_BENCHMARK("strf::to_string.reserve_calc() (strf::cv(u16sample1))")
        {
            (void)strf::to_string.reserve_calc()(strf::cv(u16sample1));
        }
        PRINT_BENCHMARK("strf::to_string.reserve(510) (strf::cv(u16sample1))")
        {
            (void)strf::to_string.reserve(510)(strf::cv(u16sample1));
        }

    }

    return 0;
}
