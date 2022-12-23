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

void allow_surrogates ()
{
    std::u16string input_utf16 {u"-----"};
    input_utf16[1] = 0xD800; // a surrogate character alone

    // convert to UTF-8
    auto str_strict = strf::to_u8string(strf::transcode(input_utf16));
    auto str_lax = strf::to_u8string
        .with(strf::surrogate_policy::lax)
        ( strf::transcode(input_utf16) );

    assert(str_strict == u8"-\uFFFD---");  // surrogate sanitized

#if defined(__cpp_char8_t)
    const char8_t str8_with_surr[] = {'-', 0xED, 0xA0, 0x80, '-', '-', '-', 0};
#else
    const char    str8_with_surr[] = "-\xED\xA0\x80---";
#endif
    assert(str_lax == str8_with_surr); // surrogate allowed
    (void) str8_with_surr;

    // now back to UTF-16
    auto utf16_strict = strf::to_u16string(strf::transcode(str_lax));

    auto utf16_lax = strf::to_u16string
        .with(strf::surrogate_policy::lax)
        ( strf::transcode(str_lax) );

    assert(utf16_strict == u"-\uFFFD\uFFFD\uFFFD---"); // surrogate sanitized
    assert(utf16_lax == input_utf16);                  // surrogate preserved

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
    allow_surrogates();
    char32();
    return 0;
}

