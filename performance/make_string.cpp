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

    char buff[100];
    PRINT_BENCHMARK("sprintf(buff, \"ten = %d, twenty = %d\", 10, 20); std::string{buff}")
    {
        sprintf(buff, "ten = %d, twenty = %d", 10, 20);
        std::string s{buff};
        (void)s;
    }

    return 0;
}
