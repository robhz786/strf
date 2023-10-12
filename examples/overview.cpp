//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/to_cfile.hpp>
#include <strf/to_string.hpp>

#if ! defined(__cpp_char8_t)

namespace strf {
constexpr auto to_u8string = to_string;
}

#endif

void sample()
{
    int value = 255;
    auto s = strf::to_string(value, " in hexadecimal is ", strf::hex(value));
    assert(s == "255 in hexadecimal is ff");

    using namespace strf::format_functions;

    auto s2 = strf::to_string(value, " in hexadecimal is ", hex(value));
    assert(s2 == "255 in hexadecimal is ff");
}

void second_example()
{
    // more formatting:  operator>(int width) : align to rigth
    //                   operator*()          : show base
    //                   p(int)               : set precision
    auto s = strf::to_string( "---"
                            , *strf::hex(255).p(4).fill(U'.') > 10
                            , "---" );
    assert(s == "---....0x00ff---");

    //
    // ranges
    //
    int array[] = {20, 30, 40};
    const char* separator = " / ";
    s = strf::to_string( "--[", strf::separated_range(array, separator), "]--");
    assert(s == "--[20 / 30 / 40]--");

    //
    // range with formatting
    //
    s = strf::to_string( "--["
                       , *strf::hex(strf::separated_range(array, separator)).p(4)
                       , "]--");
    assert(s == "--[0x0014 / 0x001e / 0x0028]--");

    // or

    s = strf::to_string( "--["
                       , *strf::fmt_separated_range(array, separator).hex().p(4)
                       , "]--");
    assert(s == "--[0x0014 / 0x001e / 0x0028]--");

    //
    // join: align a group of argument as one:
    //
    int value = 255;
    s = strf::to_string( "---"
                       , strf::join_center(30, U'.')( value
                                                    , " in hexadecimal is "
                                                    , strf::hex(value) )
                       , "---" );
    assert(s == "---...255 in hexadecimal is ff...---");


    // joins can contain any type of argument, including ranges and other joins
    s = strf::to_string( strf::join_right(30, U'.')
                           ( "{"
                           , strf::join_center(20)( "["
                                                  , strf::separated_range(array, ", ")
                                                  , "]" )
                           , "}" ));
    assert(s == "........{    [10, 20, 30]    }");
}

void format_functions()
{
    auto s = strf::to_string
        ( "---"
        , *strf::hex(255).p(4).fill(U'.') > 10
        , "---" );

    assert(s == "---....0x00ff---");
}

void reserve()
{
    auto str = strf::to_string.reserve(5000)("blah", "blah");

    assert(str == "blahblah");
    assert(str.capacity() >= 5000);
}

void basic_facet_sample()
{
    constexpr int base = 10;
    auto punct = strf::numpunct<base>{4, 3, 2}.thousands_sep(U'.');
    auto s = strf::to_string
        .with(punct)
        ("one hundred billions = ", strf::punct(100000000000LL));

    assert(s == "one hundred billions = 1.00.00.000.0000");
}

void constrained_facet()
{
    auto facet_obj = strf::constrain<std::is_signed>(strf::numpunct<10>{3});

    auto s = strf::to_string.with(facet_obj)(strf::punct(100000U), "  ", strf::punct(100000));

    assert(s == "100000  100,000");
}


void overriding_sample()
{
    auto punct_dec_1 = strf::numpunct<10>{1};
    auto punct_dec_2 = strf::numpunct<10>{2}.thousands_sep('.');
    auto punct_dec_3 = strf::numpunct<10>{3}.thousands_sep('^');

    // Below, punct_dec_3 overrides punct_dec_2, but only for signed types.
    // punct_dec_2 overrides punct_dec_1 for all input types,
    // hence the presence of punt_dec_1 bellow has no effect.

    auto s = strf::to_string
        .with( punct_dec_1
             , punct_dec_2
             , strf::constrain<std::is_signed>(punct_dec_3) )
        ( strf::punct(100000), "  ", strf::punct(100000U) ) ;

    assert(s == "100^000  10.00.00");
}


void sample_numpunct_with_alternative_encoding()
{
    // Writting in Windows-1252
    auto s = strf::to_string
        .with(strf::windows_1252<char>)
        .with(strf::numpunct<10>{4, 3, 2}.thousands_sep(0x2022))
        ("one hundred billions = ", strf::punct(100000000000LL));

    // The character U+2022 is encoded as '\225' in Windows-1252
    assert(s == "one hundred billions = 1\225""0000\225""000\225""0000");
}


void output_FILE()
{
    // writting to a FILE*
    strf::to(stdout) ("Hello World!\n");
}


void input_ouput_different_char_types()
{
    auto str = strf::to_string( strf::transcode(u"aaa-")
                              , strf::transcode(U"bbb-")
                              , strf::transcode(L"ccc") );
    assert(str ==  "aaa-bbb-ccc");
}

void input_string_encoding()
{
    // Three input string. Each one in its own character set
    auto s = strf::to_u8string( strf::transcode("\x80\xA4 -- ", strf::iso_8859_1<char>)
                              , strf::transcode("\x80\xA4 -- ", strf::iso_8859_15<char>)
                              , strf::transcode("\x80\xA4", strf::windows_1252<char>) );

    // The output by default is in UTF-8
    assert(s == u8"\u0080\u00A4 -- \u0080\u20AC -- \u20AC\u00A4");
}

void sani()
{
    // sanitize UTF-8 input
    auto s = strf::to_u8string(strf::sani("a b c \xFF d e"));
    assert(s == u8"a b c \uFFFD d e");
}

void numpunct()
{
    constexpr int base = 10;

    auto str = strf::to_string
        .with(strf::numpunct<base>{3}.thousands_sep(U'.'))
        (strf::punct(100000000000LL));

    assert(str == "100.000.000.000");
}

void variable_grouping()
{
    constexpr int base = 10;

    auto punct = strf::numpunct<base>{4, 3, 2};
    auto str = strf::to_string.with(punct)(strf::punct(100000000000LL));
    assert(str == "1,00,00,000,0000");
}

void punct_non_decimal()
{
    auto str = strf::to_string
        .with(strf::numpunct<16>{4}.thousands_sep(U'\''))
        (!strf::hex(0xffffffffffLL));

    assert(str == "ff'ffff'ffff");
}

void fast_width()
{
    const auto *str = "15.00 \xE2\x82\xAC \x80"; // "15.00 € \x80"
    auto result = strf::to_string.with(strf::fast_width)
        ( strf::right(str, 12, '*') );

    assert(result == "*15.00 \xE2\x82\xAC \x80"); // width calculated as 11
}

void width_as_fast_u32len()
{
    const auto *str = "15.00 \xE2\x82\xAC \x80"; // "15.00 € \x80"
    auto result = strf::to_string .with(strf::width_as_fast_u32len)
        ( strf::right(str, 12, '*'));
    assert(result == "****15.00 \xE2\x82\xAC \x80"); // width calculated as 8
}

void width_as_u32len()
{
    const auto *str = "15.00 \xE2\x82\xAC \x80"; // "15.00 € \x80"
    auto result = strf::to_string .with(strf::width_as_u32len) ( strf::right(str, 12, '*'));

    assert(result == "***15.00 \xE2\x82\xAC \x80"); // width calculated as 9
}

void width_in_transcode()
{
    const auto *str = "15.00 \xE2\x82\xAC \x80"; // "15.00 € \x80"

    auto res1 = strf::to_u16string.with(strf::fast_width)          (strf::transcode(str) > 12);
    auto res2 = strf::to_u16string.with(strf::width_as_fast_u32len)(strf::transcode(str) > 12);
    auto res3 = strf::to_u16string.with(strf::width_as_u32len)     (strf::transcode(str) > 12);

    assert(res1 == u" 15.00 \u20AC \uFFFD");    // width calculated as 11 ( == strlen(str) )
    assert(res2 == u"    15.00 \u20AC \uFFFD"); // width calculated as 8
    assert(res3 == u"   15.00 \u20AC \uFFFD");  // width calculated as 9
}

namespace my { // my customizations

constexpr auto my_default_facets = strf::pack
    ( strf::numpunct<10>(3)
    , strf::numpunct<16>(4).thousands_sep(U'\'') );

constexpr auto to_string = strf::to_string.with(my_default_facets);

template <typename Str>
inline auto append(Str& str)
{
    return strf::append(str).with(my_default_facets);
}

template <typename ... Args>
inline decltype(auto) to(Args&& ... args)
{
    return strf::to(std::forward<Args>(args)...).with(my_default_facets);
}

} // namespace my

void using_my_customizations()
{
    int x = 100000000;
    auto str = my::to_string(!strf::dec(x));
    assert(str == "100,000,000");

    my::append(str) (" in hexadecimal is ", *!strf::hex(x));
    assert(str == "100,000,000 in hexadecimal is 0x5f5'e100");

    char buff[500];
    my::to(buff)(!strf::dec(x), " in hexadecimal is ", *!strf::hex(x));
    assert(str == buff);

    // Overriding numpunct_c<16> back to default:
    str = my::to_string
        .with(strf::default_numpunct<16>())
        (!strf::dec(x), " in hexadecimal is ", *!strf::hex(x));
    assert(str == "100,000,000 in hexadecimal is 0x5f5e100");
}

int main()
{
    sample();
    format_functions();
    reserve();
    basic_facet_sample();
    constrained_facet();
    overriding_sample();
    output_FILE();
    input_ouput_different_char_types();
    input_string_encoding();
    sani();
    numpunct();
    variable_grouping();
    punct_non_decimal();
    fast_width();
    width_as_u32len();
    width_as_fast_u32len();
    width_in_transcode();
    using_my_customizations();
    return 0;
}
