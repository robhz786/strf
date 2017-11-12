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
#include <boost/spirit/include/karma.hpp>

using namespace boost::spirit;


int main()
{
    namespace strf = boost::stringify;
    namespace spirit = boost::spirit;
    namespace karma = boost::spirit::karma;
    char dest[1000000];
    constexpr std::size_t dest_size = sizeof(dest);


    std::cout << "\n small strings \n";

    PRINT_BENCHMARK("write_to(dest) [{\"Hello \", \"World\", \"!\"}]")
    {
        strf::write_to(dest) [{"Hello ", "World", "!"}];
    }
    PRINT_BENCHMARK("write_to(dest) (\"Hello {}!\") = {\"World\"}")
    {
        strf::write_to(dest) ("Hello {}!") = {"World"};
    }

    PRINT_BENCHMARK("karma::generate(dest, karma::lit(\"Hello \") << \"World\" << \"!\")")
    {
        char* d = dest;
        karma::generate(d, (karma::lit("Hello ") << "World" << "!"));
        *d = '\0';
    }
    PRINT_BENCHMARK("fmt_writer.write(\"Hello {}!\", \"World\")")
    {
        fmt::BasicArrayWriter<char> writer(dest, dest_size);
        writer.write("Hello {}!", "World");
    }
    PRINT_BENCHMARK("sprintf(dest, \"Hello %s!\", \"World\")")
    {
        sprintf(dest, "Hello %s!", "World");
    }

    std::cout << "\n long string ( 1000 characters ): \n";

    {
        std::string std_string_long_string(1000, 'x');
        const char* long_string = std_string_long_string.c_str();

        PRINT_BENCHMARK("write_to(dest) [{\"Hello \", long_string, \"!\"}]")
        {
            strf::write_to(dest) [{"Hello ", long_string, "!"}];
        }
        PRINT_BENCHMARK("karma::generate(dest, lit(\"Hello \") << long_string << \"!\")")
        {
            char* d = dest;
            karma::generate(d, karma::lit("Hello ") << long_string << "!");
            *d = '\0';
        }
        PRINT_BENCHMARK("fmt_writer.write(\"Hello {}!\", long_string)")
        {
            fmt::BasicArrayWriter<char> writer(dest, dest_size);
            writer.write("Hello {}!", long_string);
        }
        PRINT_BENCHMARK("sprintf(dest, \"Hello %s!\", long_string)")
        {
            sprintf(dest, "Hello %s!", long_string);
        }
    }

    std::cout << "\n integers \n";

    PRINT_BENCHMARK("write_to(dest) [{25}]")
    {
        strf::write_to(dest) [{25}];
    }
    PRINT_BENCHMARK("karma::generate(dest, karma::int_, 25)")
    {
        char* d = dest;
        karma::generate(d, karma::int_, 25);
        *d = '\0';
    }
    PRINT_BENCHMARK("fmt_writer.write(\"{}\", 25)")
    {
        fmt::BasicArrayWriter<char> writer(dest, dest_size);
        writer.write("{}", 25);
    }
    PRINT_BENCHMARK("sprintf(dest, \"%d\", 25)")
    {
        sprintf(dest, "%d", 25);
    }

    std::cout << std::endl;
    PRINT_BENCHMARK("write_to(dest) [{INT_MAX}]")
    {
        strf::write_to(dest) [{INT_MAX}];
    }
    PRINT_BENCHMARK("karma::generate(dest, karma::int_, INT_MAX);")
    {
        char* d = dest;
        karma::generate(d, karma::int_, INT_MAX);
        *d = '\0';
    }
    PRINT_BENCHMARK("fmt_writer.write(\"{}\", INT_MAX)")
    {
        fmt::BasicArrayWriter<char> writer(dest, dest_size);
        writer.write("{}", INT_MAX);
    }
    PRINT_BENCHMARK("sprintf(dest, \"%d\", INT_MAX)")
    {
        sprintf(dest, "%d", INT_MAX);
    }

    std::cout << std::endl;
    PRINT_BENCHMARK("write_to(dest) [{LLONG_MAX}]")
    {
        strf::write_to(dest) [{LLONG_MAX}];
    }
    PRINT_BENCHMARK("karma::generate(dest, karma::long_long, LLONG_MAX);")
    {
        char* d = dest;
        karma::generate(d, karma::long_long, LLONG_MAX);
        *d = '\0';
    }
    PRINT_BENCHMARK("fmt_writer.write(\"{}\", LLONG_MAX)")
    {
        fmt::BasicArrayWriter<char> writer(dest, dest_size);
        writer.write("{}", LLONG_MAX);
    }
    PRINT_BENCHMARK("sprintf(dest, \"%lld\", LLONG_MAX)")
    {
        sprintf(dest, "%lld", LLONG_MAX);
    }

    std::cout << std::endl;
    PRINT_BENCHMARK("write_to(dest) [{25, 25, 25}]")
    {
        strf::write_to(dest) [{25, 25, 25}];
    }
    PRINT_BENCHMARK("write_to(dest) (\"{}{}{}\") = {25, 25, 25}")
    {
        strf::write_to(dest) ("{}{}{}") = {25, 25, 25};
    }

    PRINT_BENCHMARK("karma::generate(dest, int_ << int_ << int_, 25, 25, 25)")
    {
        char* d = dest;
        using karma::int_;
        karma::generate(d, int_ << int_ << int_, 25, 25, 25);
        *d = '\0';
    }
    PRINT_BENCHMARK("fmt_writer.write(\"{}{}{}\", 25, 25, 25)")
    {
        fmt::BasicArrayWriter<char> writer(dest, dest_size);
        writer.write("{}{}{}", 25, 25, 25);
    }
    PRINT_BENCHMARK("sprintf(dest, \"%d%d%d\", 25, 25, 25)")
    {
        sprintf(dest, "%d%d%d", 25, 25, 25);
    }


    std::cout << std::endl;
    PRINT_BENCHMARK("write_to(dest) [{LLONG_MAX, LLONG_MAX, LLONG_MAX}]")
    {
        strf::write_to(dest) [{LLONG_MAX, LLONG_MAX, LLONG_MAX}];
    }
    PRINT_BENCHMARK("karma::generate(dest, long_long_x3, LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        char* d = dest;
        using karma::long_long;
        karma::generate(d, long_long << long_long << long_long, LLONG_MAX, LLONG_MAX, LLONG_MAX);
        *d = '\0';
    }
    PRINT_BENCHMARK("fmt_writer.write(\"{}{}{}\", LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        fmt::BasicArrayWriter<char> writer(dest, dest_size);
        writer.write("{}{}{}", LLONG_MAX, LLONG_MAX, LLONG_MAX);
    }
    PRINT_BENCHMARK("sprintf(dest, \"%d%d%d\", LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        sprintf(dest, "%lld%lld%lld", LLONG_MAX, LLONG_MAX, LLONG_MAX);
    }


    std::cout << std::endl;
    PRINT_BENCHMARK("write_to(dest) [{{25, 20}}]")
    {
        strf::write_to(dest)[{{25, 20}}];
    }
    PRINT_BENCHMARK("write_to(dest) .with(strf::width(20)) [{25}]")
    {
        strf::write_to(dest).with(strf::width(20)) [{25}];
    }
    PRINT_BENCHMARK("write_to(dest) [{ {join_right(20), {25}} }]")
    {
        strf::write_to(dest) [{ {strf::join_right(20), {25}} }];
    }
    PRINT_BENCHMARK("karma::generate(dest, right_align(20)[int_], 25);")
    {
        char* d = dest;
        using karma::right_align;
        using karma::int_;
        karma::generate(d, right_align(20)[int_], 25);
        *d = '\0';
    }
    PRINT_BENCHMARK("fmt_writer.write(\"{:20}\", 25)")
    {
        fmt::BasicArrayWriter<char> writer(dest, dest_size);
        writer.write("{:20}", 25);
    }
    PRINT_BENCHMARK("sprintf(dest, \"%20d\", 25)")
    {
        sprintf(dest, "%20d", 25);
    }


    std::cout << std::endl;
    PRINT_BENCHMARK("write_to(dest) [{{25, {6, \"<+\"}}}]")
    {
        strf::write_to(dest)[{{25, {6, "<+"}}}];
    }
    PRINT_BENCHMARK("write_to(dest).with(width(6), left, showpos) [{ 25 }]")
    {
        strf::write_to(dest).with(strf::width(6), strf::left, strf::showpos) [{ 25 }];
    }

    PRINT_BENCHMARK("write_to(dest)({strf::make_ftuple(width(6), left, showpos), {25}})")
    {
        strf::write_to(dest) [{ {strf::make_ftuple(strf::width(6), strf::left, strf::showpos), {25}} }];
    }
    PRINT_BENCHMARK("karma::generate(dest, karma::left_align(6)[dec_showpos], 25)")
    {
        char* d = dest;
        karma::int_generator<int, 10, true> dec_showpos;
        karma::generate(d, karma::right_align(6)[dec_showpos], 25);
        *d = '\0';
    }
    PRINT_BENCHMARK("fmt_writer.write(\"{:<+6}\", 25)")
    {
        fmt::BasicArrayWriter<char> writer(dest, dest_size);
        writer.write("{:<+6}", 25);
    }
    PRINT_BENCHMARK("sprintf(dest, \"%6-+d\", 25)")
    {
        sprintf(dest, "%-+6d", 25);
    }


    std::cout << std::endl;
    PRINT_BENCHMARK("write_to(dest) [{{25, \"#x\"}}]")
    {
        strf::write_to(dest) [{{25, "#x"}}];
    }
    PRINT_BENCHMARK("write_to(dest) .with(hex, showbase) [{25}]")
    {
        strf::write_to(dest).with(strf::hex, strf::showbase) [{25}];
    }
    PRINT_BENCHMARK("karma::generate(dest, karma::generate(d, lit(\"0x\") << hex, 25)")
    {
        char* d = dest;
        karma::int_generator<int, 16, false> hex;
        using karma::left_align;
        karma::generate(d, karma::lit("0x") << hex, 25);
        *d = '\0';
    }
    PRINT_BENCHMARK("fmt_writer.write(\"{:#x}\", 25)")
    {
        fmt::BasicArrayWriter<char> writer(dest, dest_size);
        writer.write("{:#x}", 25);
    }
    PRINT_BENCHMARK("sprintf(dest, \"%#x\", 25)")
    {
        sprintf(dest, "%#x", 25);
    }


    std::cout << std::endl;
    PRINT_BENCHMARK("write_to(dest) [{25, {25, {6, \"<+\"}} , {25, \"#x\"}}]")
    {
        strf::write_to(dest) [{25, {25, {6, "<+"}} , {25, "#x"}}];
    }
    PRINT_BENCHMARK("karma::generate(dest, int_ << left_6_show << \"0x\" << hex, 25,25,25)")
    {
        char* d = dest;
        karma::int_generator<int, 16, false> hex;
        karma::int_generator<int, 10, true> showpos;
        using karma::left_align;
        karma::generate(d, karma::int_ << left_align(6)[showpos] << "0x" << hex, 25,25,25);
        *d = '\0';
    }
    PRINT_BENCHMARK("fmt_writer.write(\"{}{:<6}{:#x}\", 25, 25, 25)")
    {
        fmt::BasicArrayWriter<char> writer(dest, dest_size);
        writer.write("{}{:<6}{:#x}", 25, 25, 25);
    }
    PRINT_BENCHMARK("sprintf(dest, \"%d%-+6d%#x\", 25, 25, 25)")
    {
        sprintf(dest, "%d%-+6d%#x", 25, 25, 25);
    }

    std::cout << "\n Strings and itegers mixed: \n";

    std::cout << std::endl;
    PRINT_BENCHMARK("write_to(dest) [{\"ten =  \", 10, \", twenty = \", 20}]")
    {
        strf::write_to(dest) [{"ten =  ", 10, ", twenty = ", 20}];
    }
    PRINT_BENCHMARK("karma::generate(dest, lit(\"ten= \") << int_ <<\", twenty = \" << int_, 10, 20")
    {
        char* d = dest;
        karma::generate
            ( d
            , karma::lit("ten= ") << karma::int_ << ", twenty = "<< karma::int_
            , 10, 20);

        *d = '\0';
    }
    PRINT_BENCHMARK("fmt_writer.write(\"ten = {}, twenty = {}\", 10, 20)")
    {
        fmt::BasicArrayWriter<char> writer(dest, dest_size);
        writer.write("ten = {}, twenty = {}", 10, 20);
    }
    PRINT_BENCHMARK("sprintf(dest, \"ten = %d, twenty= %d\", 10, 20)")
    {
        sprintf(dest, "ten = %d, twenty= %d", 10, 20);
    }


    return 1;
}
