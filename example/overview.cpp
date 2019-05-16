//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//[ first_example
#include <boost/stringify.hpp> // This is the only header you need to include.

namespace strf = boost::stringify::v0; // Everything is inside this namespace.
                                       // v0 is an inline namespace.
void sample()
{
    int value = 255;
    auto s = strf::to_string(value, " in hexadecimal is ", strf::hex(value));
    BOOST_ASSERT(s == "255 in hexadecimal is ff");
}
//]

void second_example()
{
    //[second_example

    namespace strf = boost::stringify::v0;

    // more formatting:  operator>(int width) : align to rigth
    //                   operator~()          : show base
    //                   p(int)               : set precision           
    auto s = strf::to_string( "---"
                            , ~strf::hex(255).p(4).fill(U'.') > 10
                            , "---" );
    BOOST_ASSERT(s == "---....0x00ff---");

    //
    // ranges
    //
    int array[] = {20, 30, 40};
    const char* separator = " / ";
    s = strf::to_string( "--[", strf::range(array, separator), "]--");
    BOOST_ASSERT(s == "--[20 / 30 / 40]--");

    //
    // range with formatting
    //
    s = strf::to_string( "--["
                       , ~strf::hex(strf::range(array, separator)).p(4)
                       , "]--");
    BOOST_ASSERT(s == "--[0x0014 / 0x001e / 0x0028]--");

    // or

    s = strf::to_string( "--["
                       , ~strf::fmt_range(array, separator).hex().p(4)
                       , "]--");
    BOOST_ASSERT(s == "--[0x0014 / 0x001e / 0x0028]--");

    //
    // join: align a group of argument as one:
    //
    int value = 255;
    s = strf::to_string( "---"
                       , strf::join_center(30, U'.')( value
                                                    , " in hexadecimal is "
                                                    , strf::hex(value) )
                       , "---" );
    BOOST_ASSERT(s == "---...255 in hexadecimal is ff...---");


    // joins can contain any type of argument, including ranges and other joins
    s = strf::to_string( strf::join_right(30, U'.')
                           ( "{"
                           , strf::join_center(20)( "["
                                                  , strf::range(array, ", ")
                                                  , "]" )
                           , "}" ));
    BOOST_ASSERT(s == "........{    [10, 20, 30]    }");
//]
}


void format_functions()
{
    //[ format_functions_example
    namespace strf = boost::stringify::v0;
    auto s = strf::to_string
        ( "---"
        , ~strf::hex(255).p(4).fill(U'.') > 10
        , "---" );

    BOOST_ASSERT(s == "---....0x00ff---");
    //]
}

void format_functions_2()
{
    //[ formatting_samples
    namespace strf = boost::stringify::v0;

    auto str = strf::to_string
        ( strf::hex(255) > 5
        , '/', strf::center(255, 7, '.').hex()
        , '/', ~strf::hex(255) % 7
        , '/', strf::multi('a', 3) ^ 7
        , '/', +strf::fmt(255) );

    BOOST_ASSERT(str == "   ff/..ff.../0x   ff/  aaa  /+255");
    //]
}

void reserve()
{
    //[ syntax_reserve
    namespace strf = boost::stringify::v0;  // v0 is an inline namespace

    auto str = strf::to_string.reserve(5000)("blah", "blah");

    BOOST_ASSERT(str == "blahblah");
    BOOST_ASSERT(str.capacity() >= 5000);
    //]
}


void sample_numpunct_with_alternative_charset()
{
//[ numpuct__with_alternative_encoding
    namespace strf = boost::stringify::v0;

    // Writting in Windows-1252
    auto s = strf::to_string
        .facets(strf::windows_1252())
        .facets(strf::str_grouping<10>{"\4\3\2"}.thousands_sep(0x2022))
        ("one hundred billions = ", 100000000000ll);

    // The character U+2022 is encoded as '\225' in Windows-1252
    BOOST_ASSERT(s == "one hundred billions = 1\2250000\225000\2250000");
//]
}



void output_FILE()
{
//[ output_FILE
    // writting to a FILE*
    namespace strf = boost::stringify::v0;
    strf::write(stdout) ("Hello World!\n");
//]
}

void input_ouput_different_char_types()
{
    //[input_output_different_char_types
    namespace strf = boost::stringify::v0;
    auto str = strf::to_string( strf::cv(u"aaa-")
                              , strf::cv(U"bbb-")
                              , strf::cv(L"ccc") );
    BOOST_ASSERT(str ==  "aaa-bbb-ccc");
    //]
}

void input_string_encoding()
{
    //[input_string_encoding
    // Three input string. Each one in its own character set
    namespace strf = boost::stringify::v0;
    auto s = strf::to_string( strf::cv("\x80\xA4 -- ", strf::iso_8859_1())
                            , strf::cv("\x80\xA4 -- ", strf::iso_8859_15())
                            , strf::cv("\x80\xA4", strf::windows_1252()) );

    // The output by default is in UTF-8
    BOOST_ASSERT(s == u8"\u0080\u00A4 -- \u0080\u20AC -- \u20AC\u00A4");
    //]
}

void sani()
{
    //[sani_utf8
    // sanitize UTF-8 input
    namespace strf = boost::stringify::v0;
    auto s = strf::to_string(strf::cv("a b c \xFF d e"));
    BOOST_ASSERT(s == u8"a b c \uFFFD d e");
    //]
}

#include <vector>

void input_ranges()
{
    //[input_range
    namespace strf = boost::stringify::v0;
    std::vector<int> v = {1, 10, 100};

    auto s = strf::to_string(strf::range(v, ", "));
    BOOST_ASSERT(s == "1, 10, 100");

    // now with formatting:
    s = strf::to_string(~strf::hex(strf::range(v, " ")));
    BOOST_ASSERT(s == "0x1 0xa 0x64");
    //]
}


void join()
{
    //[join_basic_sample
    namespace strf = boost::stringify::v0;
    std::vector<int> v = {1, 10, 100};

    auto s = strf::to_string
        ( strf::join_center(30, U'_')( '('
                                     , strf::range(v, " + ")
                                     , ") == "
                                     , std::accumulate(v.begin(), v.end(), 0) ));

    BOOST_ASSERT(s == "____(1 + 10 + 100) == 111_____");
    //]
}

int main()
{
    sample();
    format_functions();
    format_functions_2();
    reserve();
    output_FILE();
    input_ouput_different_char_types();
    input_string_encoding();
    sani();
    input_ranges();
    join();

    return 0;
}
