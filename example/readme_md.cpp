//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>

#if ! defined(__cpp_char8_t)

namespace strf {
constexpr auto to_u8string = to_string;
}

#endif

#include <cassert>
#include <iostream>
#include <strf.hpp> // The whole library is included in this header

void samples()
{
    // basic example:
    int value = 255;
    std::string s = strf::to_string(value, " in hexadecimal is ", strf::hex(value));
    assert(s == "255 in hexadecimal is ff");


    // more formatting:  operator>(int width) : align to rigth
    //                   operator~()          : show base
    //                   p(int)               : set precision
    s = strf::to_string( "---"
                       , ~strf::hex(255).p(4).fill(U'.') > 10
                       , "---" );
    assert(s == "---....0x00ff---");

    // ranges
    int array[] = {20, 30, 40};
    const char* separator = " / ";
    s = strf::to_string( "--[", strf::separated_range(array, separator), "]--");
    assert(s == "--[20 / 30 / 40]--");

    // range with formatting
    s = strf::to_string( "--["
                       , ~strf::hex(strf::separated_range(array, separator)).p(4)
                       , "]--");
    assert(s == "--[0x0014 / 0x001e / 0x0028]--");

    // join: align a group of argument as one:
    s = strf::to_string( "---"
                       , strf::join_center(30, U'.')( value
                                                    , " in hexadecimal is "
                                                    , strf::hex(value) )
                       , "---" );
    assert(s == "---...255 in hexadecimal is ff...---");

    // encoding conversion
    auto s_utf8 = strf::to_u8string( strf::cv(u"aaa-")
                                   , strf::cv(U"bbb-")
                                   , strf::cv( "\x80\xA4"
                                             , strf::windows_1252<char>() ) );
    assert(s_utf8 == u8"aaa-bbb-\u20AC\u00A4");

    // string append
    strf::assign(s) ("aaa", "bbb");
    strf::append(s) ("ccc", "ddd");
    assert(s == "aaabbbcccddd");

    // other output types
    char buff[500];
    strf::to(buff) (value, " in hexadecimal is ", strf::hex(value));
    strf::to(stdout) ("Hello, ", "World", '!');
    strf::to(std::cout.rdbuf()) ("Hello, ", "World", '!');
    std::u16string s16 = strf::to_u16string( value
                                           , u" in hexadecimal is "
                                           , strf::hex(value) );
    assert(s16 == u"255 in hexadecimal is ff");

    // alternative syntax:
    s = strf::to_string.tr("{} in hexadecimal is {}", value, strf::hex(value));
    assert(s == "255 in hexadecimal is ff");
}


int main()
{
    samples();
    return 0;
}
