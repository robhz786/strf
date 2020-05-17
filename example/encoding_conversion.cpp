//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>

#include <iostream>

#if ! defined(__cpp_char8_t)

namespace strf {
constexpr auto to_u8string = to_string;
}
using char8_t = char;

#endif


void input_ouput_different_char_types()
{
    //[input_output_different_char_types
    auto str   = strf::to_string( "aaa-"
                                , strf::conv(u"bbb-")
                                , strf::conv(U"ccc-")
                                , strf::conv(L"ddd") );

    auto str16 = strf::to_u16string( strf::conv("aaa-")
                                   , u"bbb-"
                                   , strf::conv(U"ccc-")
                                   , strf::conv(L"ddd") );

    auto str32 = strf::to_u32string( strf::conv("aaa-")
                                   , strf::conv(u"bbb-")
                                   , U"ccc-"
                                   , strf::conv(L"ddd") );

    auto wstr  = strf::to_wstring( strf::conv("aaa-")
                                 , strf::conv(u"bbb-")
                                 , strf::conv(U"ccc-")
                                 , L"ddd" );

    assert(str   ==  "aaa-bbb-ccc-ddd");
    assert(str16 == u"aaa-bbb-ccc-ddd");
    assert(str32 == U"aaa-bbb-ccc-ddd");
    assert(wstr  == L"aaa-bbb-ccc-ddd");
    //]
}

void arg()
{
    //[ arg_encoding
    auto str_utf8 = strf::to_u8string
        ( strf::conv("--\xA4--", strf::iso_8859_1<char>())
        , strf::conv("--\xA4--", strf::iso_8859_15<char>()));

    assert(str_utf8 == u8"--\u00A4----\u20AC--");
    //]
}

void allow_surrogates ()
{
    //[ allow_surrogates
    std::u16string input_utf16 {u"-----"};
    input_utf16[1] = 0xD800; // a surrogate character alone

    // convert to UTF-8
    auto str_strict = strf::to_u8string(strf::conv(input_utf16));
    auto str_lax = strf::to_u8string
        .with(strf::surrogate_policy::lax)
        ( strf::conv(input_utf16) );

    assert(str_strict == u8"-\uFFFD---");  // surrogate sanitized
    assert(str_lax == (const char8_t*)"-\xED\xA0\x80---"); // surrogate allowed

    // now back to UTF-16
    auto utf16_strict = strf::to_u16string(strf::conv(str_lax));

    auto utf16_lax = strf::to_u16string
        .with(strf::surrogate_policy::lax)
        ( strf::conv(str_lax) );

    assert(utf16_strict == u"-\uFFFD\uFFFD\uFFFD---"); // surrogate sanitized
    assert(utf16_lax == input_utf16);                  // surrogate preserved
    //]

}


int main()
{
    input_ouput_different_char_types();
    arg();
    allow_surrogates();

    return 0;
}

