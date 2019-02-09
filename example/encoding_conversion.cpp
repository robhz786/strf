//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/assert.hpp>
#include <boost/stringify.hpp>

#include <iostream>

void input_ouput_different_char_types()
{
    //[input_output_different_char_types
    namespace strf = boost::stringify::v0;

    auto str   = strf::to_string( "aaa-"
                                , strf::cv(u"bbb-")
                                , strf::cv(U"ccc-")
                                , strf::cv(L"ddd") );

    auto str16 = strf::to_u16string( strf::cv("aaa-")
                                   , u"bbb-"
                                   , strf::cv(U"ccc-")
                                   , strf::cv(L"ddd") );

    auto str32 = strf::to_u32string( strf::cv("aaa-")
                                   , strf::cv(u"bbb-")
                                   , U"ccc-"
                                   , strf::cv(L"ddd") );

    auto wstr  = strf::to_wstring( strf::cv("aaa-")
                                 , strf::cv(u"bbb-")
                                 , strf::cv(U"ccc-")
                                 , L"ddd" );

    BOOST_ASSERT(str   ==  "aaa-bbb-ccc-ddd");
    BOOST_ASSERT(str16 == u"aaa-bbb-ccc-ddd");
    BOOST_ASSERT(str32 == U"aaa-bbb-ccc-ddd");
    BOOST_ASSERT(wstr  == L"aaa-bbb-ccc-ddd");
    //]
}



void arg()
{
    //[ arg_encoding
    namespace strf = boost::stringify::v0;

    auto str_utf8 = strf::to_string
        ( strf::cv("--\xA4--", strf::iso_8859_1())
        , strf::cv("--\xA4--", strf::iso_8859_15()));

    BOOST_ASSERT(str_utf8 == u8"--\u00A4----\u20AC--");
    //]
}


void error_handling_replace()
{
    //[ error_handling_replace
    namespace strf = boost::stringify::v0;
    auto str = strf::to_string (strf::cv("--\x99--"));
    BOOST_ASSERT(str == u8"--\uFFFD--");
    //]
}

void error_signal_skip()
{
    //[ error_handling_ignore
    namespace strf = boost::stringify::v0;

    auto str = strf::to_string
        .facets(strf::encoding_policy{strf::error_handling::ignore})
        (strf::cv("--\x99--"));

    BOOST_ASSERT(str == "----");
    //]
}


void error_handling_stop()
{
    //[error_handling_stop
    namespace strf = boost::stringify::v0;

    std::error_code ec;
    try
    {
    auto str = strf::to_string
        .facets(strf::encoding_policy{strf::error_handling::stop})
        (strf::cv("--\x99--"));
    }
    catch(strf::stringify_error& x)
    {
        ec = x.code();
    }

    BOOST_ASSERT(ec == std::errc::illegal_byte_sequence);
    //]
}

void allow_surrogates ()
{
    //[ allow_surrogates
    namespace strf = boost::stringify::v0;

    std::u16string input_utf16 {u"-----"};
    input_utf16[1] = 0xD800; // a surrogate character alone

    constexpr auto allow_surrogates = strf::encoding_policy
        ( strf::error_handling::replace
        , true );

    auto str1 = strf::to_string(strf::cv(input_utf16));

    auto str2 = strf::to_string .facets(allow_surrogates) (strf::cv(input_utf16));


    BOOST_ASSERT(str1 == u8"-\uFFFD---");
    BOOST_ASSERT(str2 ==   "-\xED\xA0\x80---");

    // now back to UTF-16
    auto utf16_no_surr = strf::to_u16string(strf::cv(str2));

    auto utf16_with_surr = strf::to_u16string
        .facets(allow_surrogates)
        (strf::cv(str2));

    BOOST_ASSERT(utf16_no_surr == u"-\uFFFD\uFFFD\uFFFD---");
    BOOST_ASSERT(utf16_with_surr[1] == 0xD800);
    //]

}


int main()
{
    input_ouput_different_char_types();
    arg();
    error_handling_replace();
    error_signal_skip();
    error_handling_stop();
    allow_surrogates();

    return 0;
}

