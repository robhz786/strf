//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//[ first_example
#include <boost/stringify.hpp> // This is the only header you need to include.
#include <iostream>

namespace strf = boost::stringify::v0; // Everything is inside this namespace.
                                       // v0 is an inline namespace.

void sample()
{
    int value = 255;
    auto x = strf::to_string(value, " in hexadecimal is ", strf::hex(value));
    BOOST_ASSERT(x.value() == "255 in hexadecimal is ff");

    (void)x;
}
//]

void format_functions()
{
    //[ format_functions_example
    namespace strf = boost::stringify::v0;
    auto x = strf::to_string("---", ~strf::hex(255) > 8, "---");
    BOOST_ASSERT(x.value() == "---    0xff---");
    //]
}


void sample_numpunct()
{
//[ basic_numpuct_example
    namespace strf = boost::stringify::v0;
    constexpr int base = 10;
    auto punct = strf::str_grouping<base>{"\4\3\2"}.thousands_sep(U'.');
    auto x = strf::to_string
        .facets(punct)
        ("one hundred billions = ", *strf::fmt(100000000000ll));

    BOOST_ASSERT(x.value() == "one hundred billions = 1.00.00.000.0000");
//]
}


void output_FILE()
{
//[ output_FILE
    // writting to a FILE*
    namespace strf = boost::stringify::v0;
    auto x = strf::write(stdout) ("Hello World!\n");
//]
}

void input_ouput_different_char_types()
{
    //[input_output_different_char_types
    namespace strf = boost::stringify::v0;
    auto str16 = strf::to_u16string( strf::cv("aaa-")
                                   , u"bbb-"
                                   , strf::cv(U"ccc-")
                                   , strf::cv(L"ddd"));
    BOOST_ASSERT(str16.value() == u"aaa-bbb-ccc-ddd");
    //]
}

void windows_1252_to_utf8()
{
    //[windows_1252_to_utf8
    namespace strf = boost::stringify::v0;
    auto x = strf::to_string( strf::cv("\x80\xA4 -- ", strf::iso_8859_1())
                            , strf::cv("\x80\xA4 -- ", strf::iso_8859_15())
                            , strf::cv("\x80\xA4", strf::windows_1252()) );

    // the output is in UTF-8, unless you specify otherwise
    BOOST_ASSERT(x.value() == u8"\u0080\u00A4 -- \u0080\u20AC -- \u20AC\u00A4");
    //]
}

void sani()
{
    //[sani_utf8
    // sanitize UTF-8 input
    namespace strf = boost::stringify::v0;
    auto x = strf::to_string(strf::cv("a b c \xFF d e")).value();
    BOOST_ASSERT(x == u8"a b c \uFFFD d e");
    //]
}


int main()
{
    sample();
    format_functions();
    sample_numpunct();
    output_FILE();
    input_ouput_different_char_types();
    windows_1252_to_utf8();
    sani();

    return 0;
}
