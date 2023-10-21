//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/to_string.hpp>

#include <iostream>

#if ! defined(__cpp_char8_t)

namespace strf {
constexpr auto to_u8string = to_string;
}
using char8_t = char;

#endif


void input_ouput_different_char_types()
{
    auto str   = strf::to_string( "aaa-"
                                , strf::transcode(u"bbb-")
                                , strf::transcode(U"ccc-")
                                , strf::transcode(L"ddd") );

    auto str16 = strf::to_u16string( strf::transcode("aaa-")
                                   , u"bbb-"
                                   , strf::transcode(U"ccc-")
                                   , strf::transcode(L"ddd") );

    auto str32 = strf::to_u32string( strf::transcode("aaa-")
                                   , strf::transcode(u"bbb-")
                                   , U"ccc-"
                                   , strf::transcode(L"ddd") );

    auto wstr  = strf::to_wstring( strf::transcode("aaa-")
                                 , strf::transcode(u"bbb-")
                                 , strf::transcode(U"ccc-")
                                 , L"ddd" );

    assert(str   ==  "aaa-bbb-ccc-ddd");
    assert(str16 == u"aaa-bbb-ccc-ddd");
    assert(str32 == U"aaa-bbb-ccc-ddd");
    assert(wstr  == L"aaa-bbb-ccc-ddd");
}

void arg()
{
    auto str_utf8 = strf::to_u8string
        ( strf::transcode("--\xA4--", strf::iso_8859_1<char>)
        , strf::transcode("--\xA4--", strf::iso_8859_15<char>));

    assert(str_utf8 == u8"--\u00A4----\u20AC--");
}

void char32()
{
    char32_t ch = 0x20AC; // euro sign
    assert(strf::to_string (strf::utf<char>, ch) == "\xE2\x82\xAC");
    assert(strf::to_string (strf::iso_8859_15<char>, ch) == "\xA4");
    assert(strf::to_string (strf::iso_8859_1<char>, ch) == "?");
    (void) ch;
}

int main()
{
    input_ouput_different_char_types();
    arg();
    char32();
    return 0;
}

