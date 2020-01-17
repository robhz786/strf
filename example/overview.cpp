//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
#include <strf.hpp>

#if ! defined(__cpp_char8_t)

namespace strf {
constexpr auto to_u8string = to_string;
}

#endif

//[ first_example
#include <strf.hpp> // This is the only header you need to include.

void sample()
{
    int value = 255;
    auto s = strf::to_string(value, " in hexadecimal is ", strf::hex(value));
    assert(s == "255 in hexadecimal is ff");
}
//]

void second_example()
{
    //[second_example

    // more formatting:  operator>(int width) : align to rigth
    //                   operator~()          : show base
    //                   p(int)               : set precision
    auto s = strf::to_string( "---"
                            , ~strf::hex(255).p(4).fill(U'.') > 10
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
                       , ~strf::hex(strf::separated_range(array, separator)).p(4)
                       , "]--");
    assert(s == "--[0x0014 / 0x001e / 0x0028]--");

    // or

    s = strf::to_string( "--["
                       , ~strf::fmt_separated_range(array, separator).hex().p(4)
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
//]
}




void format_functions()
{
    //[ format_functions_example
    auto s = strf::to_string
        ( "---"
        , ~strf::hex(255).p(4).fill(U'.') > 10
        , "---" );

    assert(s == "---....0x00ff---");
    //]
}

void format_functions_2()
{
    //[ formatting_samples
    auto str = strf::to_string
        ( strf::hex(255) > 5
        , '/', strf::center(255, 7, '.').hex()
        , '/', ~strf::hex(255) % 7
        , '/', strf::multi('a', 3) ^ 7
        , '/', +strf::fmt(255) );

    assert(str == "   ff/..ff.../0x   ff/  aaa  /+255");
    //]
}

void reserve()
{
    //[ syntax_reserve
    auto str = strf::to_string.reserve(5000)("blah", "blah");

    assert(str == "blahblah");
    assert(str.capacity() >= 5000);
    //]
}

void basic_facet_sample()
{

//[ basic_facet_sample
    constexpr int base = 10;
    auto punct = strf::str_grouping<base>{"\4\3\2"}.thousands_sep(U'.');
    auto s = strf::to_string
        .with(punct)
        ("one hundred billions = ", 100000000000ll);

    assert(s == "one hundred billions = 1.00.00.000.0000");
//]
}


void constrained_facet()
{
    //[ constrained_facet_sample
    auto facet_obj = strf::constrain<std::is_signed>(strf::monotonic_grouping<10>{3});

    auto s = strf::to_string.with(facet_obj)(100000u, "  ", 100000);

    assert(s == "100000  100,000");
    //]
}


void overriding_sample()
{
    //[ facets_overriding
    auto punct_dec_1 = strf::monotonic_grouping<10>{1};
    auto punct_dec_2 = strf::monotonic_grouping<10>{2}.thousands_sep('.');
    auto punct_dec_3 = strf::monotonic_grouping<10>{3}.thousands_sep('^');;

    // Below, punct_dec_3 overrides punct_dec_2, but only for signed types.
    // punct_dec_2 overrides punct_dec_1 for all input types,
    // hence the presence of punt_dec_1 bellow has no effect.

    auto s = strf::to_string
        .with( punct_dec_1
               , punct_dec_2
               , strf::constrain<std::is_signed>(punct_dec_3) )
        ( 100000, "  ", 100000u ) ;

    assert(s == "100^000  10.00.00");
    //]
}


void get_facet_sample()
{
    //[ get_facet_sample
    auto punct_hex  = strf::monotonic_grouping<16>{4}.thousands_sep('\'');
    auto punct_dec  = strf::monotonic_grouping<10>{3}.thousands_sep('.');

    auto fp = strf::pack
        ( std::ref(punct_hex) // note the use of std::ref here
        , strf::constrain<strf::is_int_number>(std::ref(punct_dec)) );//and here

    decltype(auto) f1 = strf::get_facet<strf::numpunct_c<16>, int>(fp);
    assert(&f1 == &punct_hex);

    decltype(auto) f2 = strf::get_facet<strf::numpunct_c<10>, int>(fp);
    assert(&f2 == &punct_dec);

    decltype(auto) f3 = strf::get_facet<strf::numpunct_c<10>, double>(fp);
    assert(&f3 == &strf::numpunct_c<10>::get_default());
    //]
    (void)f1;
    (void)f2;
    (void)f3;
}


void sample_numpunct_with_alternative_charset()
{
//[ numpuct__with_alternative_encoding
    // Writting in Windows-1252
    auto s = strf::to_string
        .with(strf::windows_1252<char>())
        .with(strf::str_grouping<10>{"\4\3\2"}.thousands_sep(0x2022))
        ("one hundred billions = ", 100000000000ll);

    // The character U+2022 is encoded as '\225' in Windows-1252
    assert(s == "one hundred billions = 1\225""0000\225""000\225""0000");
//]
}


void output_FILE()
{
//[ output_FILE
    // writting to a FILE*
    strf::to(stdout) ("Hello World!\n");
//]
}

void input_ouput_different_char_types()
{
    //[input_output_different_char_types
    auto str = strf::to_string( strf::cv(u"aaa-")
                              , strf::cv(U"bbb-")
                              , strf::cv(L"ccc") );
    assert(str ==  "aaa-bbb-ccc");
    //]
}

void input_string_encoding()
{
    //[input_string_encoding
    // Three input string. Each one in its own character set
    auto s = strf::to_u8string( strf::cv("\x80\xA4 -- ", strf::iso_8859_1<char>())
                              , strf::cv("\x80\xA4 -- ", strf::iso_8859_15<char>())
                              , strf::cv("\x80\xA4", strf::windows_1252<char>()) );

    // The output by default is in UTF-8
    assert(s == u8"\u0080\u00A4 -- \u0080\u20AC -- \u20AC\u00A4");
    //]
}

void sani()
{
    //[sani_utf8
    // sanitize UTF-8 input
    auto s = strf::to_u8string(strf::sani("a b c \xFF d e"));
    assert(s == u8"a b c \uFFFD d e");
    //]
}

void monotonic_grouping()
{
    //[monotonic_grouping
    constexpr int base = 10;

    auto str = strf::to_string
        .with(strf::monotonic_grouping<base>{3}.thousands_sep(U'.'))
        (100000000000ll);

    assert(str == "100.000.000.000");
    //]
}

void str_grouping()
{
    //[str_grouping
    constexpr int base = 10;

    auto punct = strf::str_grouping<base>{"\4\3\2"};
    auto str = strf::to_string.with(punct)(100000000000ll);
    assert(str == "1,00,00,000,0000");
    //]
}

void punct_non_decimal()
{
    //[punct_non_decimal
    auto str = strf::to_string
        .with(strf::monotonic_grouping<16>{4}.thousands_sep(U'\''))
        (strf::hex(0xffffffffffLL));

    assert(str == "ff'ffff'ffff");
    //]
}

// void width_as_u32len()
// {
//     //[width_as_u32len
//     //     auto str = strf::to_u8string
//         .with(strf::width_as_u32len<char8_t>{})
//         (strf::right(u8"áéíóú", 12, U'.'));

//     assert(str == u8".......áéíóú");
//     //]
// }

// void width_func()
// {
//     //[width_func
//     auto my_width_calculator =
//         [] (int limit, const char32_t* it, const char32_t* end)
//     {
//         int sum = 0;
//         for (; sum < limit && it != end; ++it)
//         {
//             auto ch = *it;
//             sum += ((0x2E80 <= ch && ch <= 0x9FFF) ? 2 : 1);
//         }
//         return sum;
//     };

//     auto str = strf::to_u8string
//         .with(strf::width_as(my_width_calculator))
//         (strf::right(u8"今晩は", 10, U'.'));

//     assert(str == u8"....今晩は");
//     //]
// }

//[avoid_repetitions
namespace my { // my customizations

const auto my_default_facets = strf::pack
    ( strf::monotonic_grouping<10>(3)
    , strf::monotonic_grouping<16>(4).thousands_sep(U'\'')
    , strf::surrogate_policy::lax );

const auto to_string = strf::to_string.with(my_default_facets);

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
    auto str = my::to_string(x);
    assert(str == "100,000,000");

    my::append(str) (" in hexadecimal is ", ~strf::hex(x));
    assert(str == "100,000,000 in hexadecimal is 0x5f5'e100");

    char buff[500];
    my::to(buff)(x, " in hexadecimal is ", ~strf::hex(x));
    assert(str == buff);

    // Overriding numpunct_c<16> back to default:
    str = my::to_string
        .with(strf::default_numpunct<16>())
        (x, " in hexadecimal is ", ~strf::hex(x));
    assert(str == "100,000,000 in hexadecimal is 0x5f5e100");
}
//]

int main()
{
    sample();
    format_functions();
    format_functions_2();
    reserve();
    basic_facet_sample();
    constrained_facet();
    overriding_sample();
    get_facet_sample();
    output_FILE();
    input_ouput_different_char_types();
    input_string_encoding();
    sani();
    monotonic_grouping();
    str_grouping();
    punct_non_decimal();
    // width_as_u32len();
    // width_func();
    using_my_customizations();
    return 0;
}
