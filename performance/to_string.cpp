//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <sstream>
#include <strf.hpp>
#include "loop_timer.hpp"
#include "fmt/format.h"
#include <climits>

int main()
{
    PRINT_BENCHMARK("strf::to_string.reserve_calc() (\"Hello \", \"World\", \"!\")")
    {
        auto str = strf::to_string.reserve_calc()("Hello ", "World", "!");
        escape(str.data());
    }
    PRINT_BENCHMARK("strf::to_string.reserve(12)    (\"Hello \", \"World\", \"!\")")
    {
        auto str = strf::to_string.reserve(12)("Hello ", "World", "!");
        escape(str.data());
    }
    PRINT_BENCHMARK("strf::to_string.no_reserve()   (\"Hello \", \"World\", \"!\")")
    {
        auto str = strf::to_string.no_reserve()("Hello ", "World", "!");
        escape(str.data());
    }
    PRINT_BENCHMARK("strf::to_string.reserve_calc() .tr(\"Hello {}!\", \"World\")")
    {
        auto str = strf::to_string.reserve_calc().tr("Hello {}!", "World");
        escape(str.data());
    }
    PRINT_BENCHMARK("strf::to_string.reserve(12)    .tr(\"Hello {}!\", \"World\")")
    {
        auto str = strf::to_string.reserve(12).tr("Hello {}!", "World");
        escape(str.data());
    }
    PRINT_BENCHMARK("strf::to_string.no_reserve()   .tr(\"Hello {}!\", \"World\")")
    {
        auto str = strf::to_string.no_reserve().tr("Hello {}!", "World");
        escape(str.data());
    }
    PRINT_BENCHMARK("fmt::format(\"Hello {}!\", \"World\")")
    {
        auto str = fmt::format("Hello {}!", "World");
        escape(str.data());
    }

    std::cout << "\n";

    PRINT_BENCHMARK("strf::to_string.reserve_calc() (25)")
    {
        auto str = strf::to_string.reserve_calc()(25);
        escape(str.data());
    }
    PRINT_BENCHMARK("strf::to_string.reserve(2)     (25)")
    {
        auto str = strf::to_string.reserve(2)(25);
        escape(str.data());
    }
    PRINT_BENCHMARK("strf::to_string.no_reserve()   (25)")
    {
        auto str = strf::to_string.no_reserve()(25);
        escape(str.data());
    }
    PRINT_BENCHMARK("fmt::format(\"{}\", 25)")
    {
        auto str = fmt::format("{}", 25);
        escape(str.data());
    }
    PRINT_BENCHMARK("std::to_string(25)")
    {
        auto str = std::to_string(25);
        escape(str.data());
    }

    std::cout << "\n";

    PRINT_BENCHMARK("strf::to_string.reserve_calc() (LLONG_MAX)")
    {
        auto str = strf::to_string.reserve_calc()(LLONG_MAX);
        escape(str.data());
    }
    PRINT_BENCHMARK("strf::to_string.reserve(20)    (LLONG_MAX)")
    {
        auto str = strf::to_string.reserve(20) (LLONG_MAX);
        escape(str.data());
    }
    PRINT_BENCHMARK("strf::to_string.no_reserve()   (LLONG_MAX)")
    {
        auto str = strf::to_string.no_reserve() (LLONG_MAX);
        escape(str.data());
    }
    PRINT_BENCHMARK("fmt::format(\"{}\", LLONG_MAX)")
    {
        auto str = fmt::format("{}", LLONG_MAX);
        escape(str.data());
    }
    PRINT_BENCHMARK("std::to_string(LLONG_MAX)")
    {
        auto str = std::to_string(LLONG_MAX);
        escape(str.data());
    }

    std::cout << "\n";

    PRINT_BENCHMARK("strf::to_string .reserve_calc() (strf::right(\"aa\", 20))")
    {
        auto str = strf::to_string .reserve_calc() (strf::right("aa", 20));
        escape(str.data());
    }
    PRINT_BENCHMARK("strf::to_string .reserve(20)    (strf::right(\"aa\", 20))")
    {
        auto str = strf::to_string .reserve(20) (strf::right("aa", 20));
        escape(str.data());
    }
    PRINT_BENCHMARK("strf::to_string .no_reserve()   (strf::right(\"aa\", 20))")
    {
        auto str = strf::to_string .no_reserve() (strf::right("aa", 20));
        escape(str.data());
    }
    PRINT_BENCHMARK("fmt::format(dest, \"{:20}\", \"aa\")")
    {
        auto str = fmt::format("{:20}", "aa");
        escape(str.data());
    }

    std::cout << "\n";

    PRINT_BENCHMARK("strf::to_string .reserve_calc() (\"ten = \", 10, \", twenty = \", 20)")
    {
        auto str = strf::to_string .reserve_calc() ("ten = ", 10, ", twenty = ", 20);
        escape(str.data());
    }
    PRINT_BENCHMARK("strf::to_string .reserve(21)    (\"ten = \", 10, \", twenty = \", 20)")
    {
        auto str = strf::to_string .reserve(21) ("ten = ", 10, ", twenty = ", 20);
        escape(str.data());
    }
    PRINT_BENCHMARK("strf::to_string .no_reserve()   (\"ten = \", 10, \", twenty = \", 20)")
    {
        auto str = strf::to_string .no_reserve() ("ten = ", 10, ", twenty = ", 20);
        escape(str.data());
    }
    PRINT_BENCHMARK("strf::to_string .reserve_calc() .tr(\"ten = {}, twenty = {}\", 10, 20)")
    {
        auto str = strf::to_string .reserve_calc() .tr("ten = {}, twenty = {}", 10, 20);
        escape(str.data());
    }
    PRINT_BENCHMARK("strf::to_string .reserve(21)    .tr(\"ten = {}, twenty = {}\", 10, 20)")
    {
        auto str = strf::to_string .reserve(21).tr("ten = {}, twenty = {}", 10, 20);
        escape(str.data());
    }
    PRINT_BENCHMARK("strf::to_string .no_reserve()   .tr(\"ten = {}, twenty = {}\", 10, 20)")
    {
        auto str = strf::to_string .no_reserve() .tr("ten = {}, twenty = {}", 10, 20);
        escape(str.data());
    }
    PRINT_BENCHMARK("fmt::format(\"ten = {}, twenty = {}\", 10, 20)")
    {
        auto str = fmt::format("ten = {}, twenty = {}", 10, 20);
        escape(str.data());
    }

    PRINT_BENCHMARK("oss << \"ten = \" << 10 << \", twenty = \" << 20")
    {
        std::ostringstream oss;
        oss << "ten = " << 10 << ", twenty = " << 20;
        escape(oss);
    }
    PRINT_BENCHMARK("oss << \"ten = \" << 10 << \", twenty = \" << 20 ; auto s = oss.str()")
    {
        std::ostringstream oss;
        oss << "ten = " << 10 << ", twenty = " << 20;
        std::string str = oss.str();
        escape(str.data());
    }

    {
        char buff[200];
        PRINT_BENCHMARK("sprintf(buff, \"ten = %d, twenty = %d\", 10, 20); std::string{buff}")
        {
            sprintf(buff, "ten = %d, twenty = %d", 10, 20);
            std::string str{buff};
            escape(str.data());
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
            auto str = strf::to_u16string(strf::cv(u8sample1));
            escape(str.data());
        }
        PRINT_BENCHMARK("strf::write(buff) (strf::cv(u8sample1)); strf::to_u16string.reserve_calc() (buff)")
        {
            (void)strf::write(buff)(strf::cv(u8sample1));
            auto str = strf::to_u16string.reserve_calc() (buff);
            escape(str.data());
        }
        PRINT_BENCHMARK("strf::to_u16string.reserve_calc() (strf::cv(u8sample1))")
        {
            auto str = strf::to_u16string.reserve_calc()(strf::cv(u8sample1));
            escape(str.data());
        }
        PRINT_BENCHMARK("strf::to_u16string .reserve(510) (strf::cv(u8sample1))")
        {
            auto str = strf::to_u16string.reserve(510)(strf::cv(u8sample1));
            escape(str.data());
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
            auto str = strf::to_string(strf::cv(u16sample1));
            escape(str.data());
        }
        PRINT_BENCHMARK("strf::write(buff) (strf::cv(u16sample1)); strf::to_string.reserve_calc()(buff)")
        {
            (void)strf::write(buff)(strf::cv(u16sample1));
            auto str = strf::to_string.reserve_calc()(buff);
            escape(str.data());
        }
        PRINT_BENCHMARK("strf::to_string.reserve_calc() (strf::cv(u16sample1))")
        {
            auto str = strf::to_string.reserve_calc()(strf::cv(u16sample1));
            escape(str.data());
        }
        PRINT_BENCHMARK("strf::to_string.reserve(510) (strf::cv(u16sample1))")
        {
            auto str = strf::to_string.reserve(510)(strf::cv(u16sample1));
            escape(str.data());
        }
    }

    return 0;
}
