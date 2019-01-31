//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//[ first_example
#include <boost/stringify.hpp> // This is the only header you need to include.
#include <iostream>
#include <numeric>

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
        ("one hundred billions = ", 100000000000ll);

    BOOST_ASSERT(x.value() == "one hundred billions = 1.00.00.000.0000");
//]
}


void sample_numpunct_with_alternative_charset()
{
//[ numpuct__with_alternative_encoding
    namespace strf = boost::stringify::v0;

    // Writting in Windows-1252
    auto x = strf::to_string
        .facets(strf::windows_1252())
        .facets(strf::str_grouping<10>{"\4\3\2"}.thousands_sep(0x2022))
        ("one hundred billions = ", 100000000000ll);

    // The character U+2022 is encoded as '\225' in Windows-1252
    BOOST_ASSERT(x.value() == "one hundred billions = 1\2250000\225000\2250000");
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
    auto str = strf::to_string( strf::cv(u"aaa-")
                              , strf::cv(U"bbb-")
                              , strf::cv(L"ccc") );
    BOOST_ASSERT(str.value() ==  "aaa-bbb-ccc");
    //]
}

void input_string_encoding()
{
    //[input_string_encoding
    // Three input string. Each one in its own character set
    namespace strf = boost::stringify::v0;
    auto x = strf::to_string( strf::cv("\x80\xA4 -- ", strf::iso_8859_1())
                            , strf::cv("\x80\xA4 -- ", strf::iso_8859_15())
                            , strf::cv("\x80\xA4", strf::windows_1252()) );

    // The output by default is in UTF-8
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

#include <vector>

void input_ranges()
{
    //[input_range
    namespace strf = boost::stringify::v0;
    std::vector<int> v = {1, 10, 100};

    auto x = strf::to_string(strf::range(v, ", ")).value();
    BOOST_ASSERT(x == "1, 10, 100");

    // now with formatting:
    x = strf::to_string(~strf::hex(strf::range(v, " "))).value();
    BOOST_ASSERT(x == "0x1 0xa 0x64");
    //]
}


void join()
{
    //[join_basic_sample
    namespace strf = boost::stringify::v0;
    std::vector<int> v = {1, 10, 100};

    auto x = strf::to_string
        ( strf::join_center(30, U'_')( '('
                                     , strf::range(v, " + ")
                                     , ") == "
                                     , std::accumulate(v.begin(), v.end(), 0) ));

    BOOST_ASSERT(x.value() == "____(1 + 10 + 100) == 111_____");
    //]
}

int main()
{
    sample();
    format_functions();
    sample_numpunct();
    output_FILE();
    input_ouput_different_char_types();
    input_string_encoding();
    sani();
    input_ranges();
    join();

    return 0;
}
