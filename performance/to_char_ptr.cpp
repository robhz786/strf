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

    PRINT_BENCHMARK("strf::write_to(dest) = {\"Hello \", \"World\", \"!\"}")
    {
        auto err = strf::write_to(dest) = {"Hello ", "World", "!"};
        (void)err;
    }
    PRINT_BENCHMARK("strf::write_to(dest) [\"Hello {}!\"] = {\"World\"}")
    {
        auto err = strf::write_to(dest) ["Hello {}!"] = {"World"};
        (void)err;
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

        PRINT_BENCHMARK("strf::write_to(dest) = {\"Hello \", long_string, \"!\"}")
        {
            auto err = strf::write_to(dest) = {"Hello ", long_string, "!"};
            (void)err;
        }
        PRINT_BENCHMARK("strf::write_to(dest) [\"Hello {}!\"] = {long_string}")
        {
            auto err = strf::write_to(dest) ["Hello {}!"] = {long_string};
            (void)err;
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

    std::cout << "\n padding \n";

    PRINT_BENCHMARK("strf::write_to(dest) = {strf::right(\"aa\", 20)}")
    {
        auto err = strf::write_to(dest) = {strf::right("aa", 20)};
        (void)err;
    }
    PRINT_BENCHMARK("strf::write_to(dest) = { {join_right(20), {\"aa\"}} }")
    {
        auto err = strf::write_to(dest) = { {strf::join_right(20), {"aa"}} };
        (void)err;
    }
    PRINT_BENCHMARK("karma::generate(dest, karma::right_align(20)[\"aa\"])")
    {
        char* d = dest;
        karma::generate(d, karma::right_align(20)["aa"]);
        *d = '\0';
    }
    PRINT_BENCHMARK("fmt_writer.write(dest, \"{:20}\", \"aa\")")
    {
        fmt::BasicArrayWriter<char> writer(dest, dest_size);
        writer.write(dest, "{:20}", "aa");
    }
    PRINT_BENCHMARK("sprintf(dest, \"%20s\", \"aa\")")
    {
        sprintf(dest, "%20s", "aa");
    }

    std::cout << "\n integers \n";

    PRINT_BENCHMARK("strf::write_to(dest) = {25}")
    {
        auto err = strf::write_to(dest) = {25};
        (void)err;
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
    PRINT_BENCHMARK("strf::write_to(dest) = {LLONG_MAX}")
    {
        auto err = strf::write_to(dest) = {LLONG_MAX};
        (void)err;
    }
    PRINT_BENCHMARK("karma::generate(dest, karma::long_long, LLONG_MAX)")
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
    strf::monotonic_grouping<10> numpunct_3(3);
    PRINT_BENCHMARK("strf::write_to(dest).with(numpunct_3) = {LLONG_MAX}")
    {
        auto err = strf::write_to(dest).with(numpunct_3) = {LLONG_MAX};
        (void)err;
    }

    
/*
    std::cout << std::endl;
    PRINT_BENCHMARK("strf::write_to(dest) = {25, 25, 25}")
    {
        auto err = strf::write_to(dest) = {25, 25, 25};
        (void)err;
    }
    PRINT_BENCHMARK("strf::write_to(dest) [\"{}{}{}\"] = {25, 25, 25}")
    {
        auto err = strf::write_to(dest) ["{}{}{}"] = {25, 25, 25};
        (void)err;
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
*/

    std::cout << std::endl;
    PRINT_BENCHMARK("strf::write_to(dest) = {LLONG_MAX, LLONG_MAX, LLONG_MAX}")
    {
        auto err = strf::write_to(dest) = {LLONG_MAX, LLONG_MAX, LLONG_MAX};
        (void)err;
    }
    PRINT_BENCHMARK("strf::write_to(dest) [\"{}{}{}\"] = {LLONG_MAX, LLONG_MAX, LLONG_MAX}")
    {
        auto err = strf::write_to(dest) ["{}{}{}"] = {LLONG_MAX, LLONG_MAX, LLONG_MAX};
        (void)err;
    }
    PRINT_BENCHMARK("karma::generate(d, long_long << long_long << long_long, LLONG_MAX, LLONG_MAX, LLONG_MAX);")
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

    std::cout << "\n formatted integers \n";
    PRINT_BENCHMARK("strf::write_to(dest) [\"{}{}{}\"] = {55555, +strf::left(55555, 8) , ~strf::hex(55555)}")
    {
        auto err = strf::write_to(dest) ["{}{}{}"] = {55555, +strf::left(55555, 8) , ~strf::hex(55555)};
        (void)err;
    }
    PRINT_BENCHMARK("strf::write_to(dest) = {55555, +strf::left(55555, 8) , ~strf::hex(55555)}")
    {
        auto err = strf::write_to(dest) = {55555, +strf::left(55555, 8) , ~strf::hex(55555)};
        (void)err;
    }


    PRINT_BENCHMARK("karma::generate(dest, int_ << left_align(8)[int_generator<int, 10, true>{}] << \"0x\" << int_generator<int, 16, false>{}, 55555, 55555, 55555)")
    {
        char* d = dest;
        using karma::int_generator;
        using karma::left_align;
        karma::generate(d, int_ << left_align(8)[int_generator<int, 10, true>{}] << "0x" << int_generator<int, 16, false>{}, 55555, 55555, 55555);

        *d = '\0';
    }
    PRINT_BENCHMARK("fmt_writer.write(\"{}{:<8}{:#x}\", 55555, 55555, 55555)")
    {
        fmt::BasicArrayWriter<char> writer(dest, dest_size);
        writer.write("{}{:<8}{:#x}", 55555, 55555, 55555);
    }
    PRINT_BENCHMARK("sprintf(dest, \"%d%-+8d%#x\", 55555, 55555, 55555)")
    {
        sprintf(dest, "%d%-+8d%#x", 55555, 55555, 55555);
    }

    std::cout << "\n Strings and itegers mixed: \n";

    PRINT_BENCHMARK("strf::write_to(dest) [\"blah blah {} blah {} blah {}\"] = {INT_MAX, {1234, ~strf::hex(1234)<8, \"abcdef\"}")
    {
        auto err = strf::write_to(dest) ["blah blah {} blah {} blah {}"] = {INT_MAX, ~strf::hex(1234)<8, "abcdef"};
        (void)err;
    }
    PRINT_BENCHMARK("karma::generate(dest, lit(\"blah blah \") << int_ << \" blah \" << left_align(8)[int_generator<int, 16, false>{}] << \" blah \" << \"abcdef\", INT_MAX, 1234)")
    {
        char* d = dest;
        using namespace karma;
        karma::generate
            ( d
              , lit("blah blah ") << int_ << " blah " << left_align(8)[int_generator<int, 16, false>{}] << " blah " << "abcdef"
            , INT_MAX, 1234
            );

        *d = '\0';
    }
    PRINT_BENCHMARK("fmt_writer.write(\"blah blah {} blah {:<#8x} blah {}\", INT_MAX, 1234, \"abcdef\")")
    {
        fmt::BasicArrayWriter<char> writer(dest, dest_size);
        writer.write("blah blah {} blah {:<#8x} blah {}", INT_MAX, 1234, "abcdef");
    }
    PRINT_BENCHMARK("sprintf(dest, \"blah blah %d blah %#-8x blah %s\", INT_MAX, 1234, \"abcdef\")")
    {
        sprintf(dest, "blah blah %d blah %#-8x blah %s", INT_MAX, 1234, "abcdef");
    }

    std::cout << std::endl;
    PRINT_BENCHMARK("strf::write_to(dest) [\"ten = {}, twenty = {}\"] = {10, 20}")
    {
        auto err = strf::write_to(dest) ["ten = {}, twenty = {}"] = {10, 20};
        (void)err;
    }
    PRINT_BENCHMARK("strf::write_to(dest) = {\"ten =  \", 10, \", twenty = \", 20}")
    {
        auto err = strf::write_to(dest) = {"ten =  ", 10, ", twenty = ", 20};
        (void)err;
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
