//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
#include <boost/stringify.hpp>

#if ! defined(__cpp_char8_t)

namespace boost{ namespace stringify{ inline namespace v0{
constexpr auto to_u8string = to_string;
}}}

#endif

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

void leading_expression_exception()
{
    //[leading_expression_error_handling
    namespace strf = boost::stringify::v0;
    char small_buff[15];

    // success
    auto ec = strf::ec_write(small_buff) ("twenty = ", 20);
    BOOST_ASSERT(ec == std::error_code{});

    // success
    strf::write(small_buff) ("ten = ", 10);
    BOOST_ASSERT(0 == strcmp(small_buff, "ten = 10"));

    // failure
    ec = strf::ec_write(small_buff) ("ten = ", 10, ", twenty = ", 20);
    BOOST_ASSERT(ec == std::errc::result_out_of_range);

    // failure
    ec = std::error_code{};
    try
    {
        strf::write(small_buff) ("ten = ", 10, ", twenty = ", 20);
    }
    catch(strf::stringify_error& e)
    {
        ec = e.code();
    }
    BOOST_ASSERT(ec == std::errc::result_out_of_range);
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

void basic_facet_sample()
{

//[ basic_facet_sample
    namespace strf = boost::stringify::v0;
    constexpr int base = 10;
    auto punct = strf::str_grouping<base>{"\4\3\2"}.thousands_sep(U'.');
    auto s = strf::to_string
        .facets(punct)
        ("one hundred billions = ", 100000000000ll);

    BOOST_ASSERT(s == "one hundred billions = 1.00.00.000.0000");
//]
}


void constrained_facet()
{
    //[ constrained_facet_sample

    namespace strf = boost::stringify::v0;

    auto facet_obj = strf::constrain<std::is_signed>(strf::monotonic_grouping<10>{3});

    auto s = strf::to_string.facets(facet_obj)(100000u, "  ", 100000);

    BOOST_ASSERT(s == "100000  100,000");
    //]
}


void overriding_sample()
{
    //[ facets_overriding
    namespace strf = boost::stringify::v0;

    auto punct_dec_1 = strf::monotonic_grouping<10>{1};
    auto punct_dec_2 = strf::monotonic_grouping<10>{2}.thousands_sep('.');
    auto punct_dec_3 = strf::monotonic_grouping<10>{3}.thousands_sep('^');;

    // Below, punct_dec_3 overrides punct_dec_2, but only for signed types.
    // punct_dec_2 overrides punct_dec_1 for all input types,
    // hence the presence of punt_dec_1 bellow has no effect.

    auto s = strf::to_string
        .facets( punct_dec_1
               , punct_dec_2
               , strf::constrain<std::is_signed>(punct_dec_3) )
        ( 100000, "  ", 100000u ) ;

    BOOST_ASSERT(s == "100^000  10.00.00");
    //]
}


void get_facet_sample()
{
    //[ get_facet_sample
    namespace strf = boost::stringify::v0;

    auto punct_hex  = strf::monotonic_grouping<16>{4}.thousands_sep('\'');
    auto punct_dec  = strf::monotonic_grouping<10>{3}.thousands_sep('.');

    auto fp = strf::pack
        ( std::ref(punct_hex) // note the use of std::ref here
        , strf::constrain<strf::is_int_number>(std::ref(punct_dec)) );//and here

    decltype(auto) f1 = strf::get_facet<strf::numpunct_c<16>, int>(fp);
    BOOST_ASSERT(&f1 == &punct_hex);

    decltype(auto) f2 = strf::get_facet<strf::numpunct_c<10>, int>(fp);
    BOOST_ASSERT(&f2 == &punct_dec);

    decltype(auto) f3 = strf::get_facet<strf::numpunct_c<10>, double>(fp);
    BOOST_ASSERT(&f3 == &strf::numpunct_c<10>::get_default());
    //]
    (void)f1;
    (void)f2;
    (void)f3;
}


void sample_numpunct_with_alternative_charset()
{
//[ numpuct__with_alternative_encoding
    namespace strf = boost::stringify::v0;

    // Writting in Windows-1252
    auto s = strf::to_string
        .facets(strf::windows_1252<char>())
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
    auto s = strf::to_u8string( strf::cv("\x80\xA4 -- ", strf::iso_8859_1<char>())
                              , strf::cv("\x80\xA4 -- ", strf::iso_8859_15<char>())
                              , strf::cv("\x80\xA4", strf::windows_1252<char>()) );

    // The output by default is in UTF-8
    BOOST_ASSERT(s == u8"\u0080\u00A4 -- \u0080\u20AC -- \u20AC\u00A4");
    //]
}

void sani()
{
    //[sani_utf8
    // sanitize UTF-8 input
    namespace strf = boost::stringify::v0;
    auto s = strf::to_u8string(strf::cv("a b c \xFF d e"));
    BOOST_ASSERT(s == u8"a b c \uFFFD d e");
    //]
}

void monotonic_grouping()
{
    //[monotonic_grouping
    namespace strf = boost::stringify::v0;
    constexpr int base = 10;

    auto str = strf::to_string
        .facets(strf::monotonic_grouping<base>{3}.thousands_sep(U'.'))
        (100000000000ll);

    BOOST_ASSERT(str == "100.000.000.000");
    //]
}

void str_grouping()
{
    //[str_grouping
    namespace strf = boost::stringify::v0;
    constexpr int base = 10;

    auto punct = strf::str_grouping<base>{"\4\3\2"};
    auto str = strf::to_string.facets(punct)(100000000000ll);
    BOOST_ASSERT(str == "1,00,00,000,0000");
    //]
}

void punct_non_decimal()
{
    //[punct_non_decimal
    namespace strf = boost::stringify::v0;
    auto str = strf::to_string
        .facets(strf::monotonic_grouping<16>{4}.thousands_sep(U'\''))
        (strf::hex(0xffffffffffLL));

    BOOST_ASSERT(str == "ff'ffff'ffff");
    //]
}
void width_as_len()
{
    //[width_as_len
    namespace strf = boost::stringify::v0;

    auto str = strf::to_u8string
        .facets(strf::width_as_len())
        (strf::right(u8"áéíóú", 12, U'.'));

    BOOST_ASSERT(str == u8"..áéíóú");
    //]
}
void width_as_u32len()
{
    //[width_as_u32len
    namespace strf = boost::stringify::v0;

    auto str = strf::to_u8string
        .facets(strf::width_as_u32len())
        (strf::right(u8"áéíóú", 12, U'.'));

    BOOST_ASSERT(str == u8".......áéíóú");
    //]
}

void width_func()
{
    //[width_func
    auto my_width_calculator =
        [] (int limit, const char32_t* it, const char32_t* end)
    {
        int sum = 0;
        for (; sum < limit && it != end; ++it)
        {
            auto ch = *it;
            sum += ((0x2E80 <= ch && ch <= 0x9FFF) ? 2 : 1);
        }
        return sum;
    };

    auto str = strf::to_u8string
        .facets(strf::width_as(my_width_calculator))
        (strf::right(u8"今晩は", 10, U'.'));

    BOOST_ASSERT(str == u8"....今晩は");
    //]
}

//[avoid_repetitions

namespace my { // my customizations

namespace strf = boost::stringify::v0;

const auto my_default_facets = strf::pack
    ( strf::monotonic_grouping<10>(3)
    , strf::monotonic_grouping<16>(4).thousands_sep(U'\'')
    , strf::width_as_u32len()
    , strf::surrogate_policy::lax
    , strf::encoding_error::stop );

const auto to_string = strf::to_string.facets(my_default_facets);

template <typename Str>
inline auto append(Str& str)
{
    return strf::append(str).facets(my_default_facets);
}

template <typename ... Args>
inline decltype(auto) write(Args&& ... args)
{
    return strf::write(std::forward<Args>(args)...).facets(my_default_facets);
}

} // namespace my

void using_my_customizations()
{
    namespace strf = boost::stringify::v0;

    int x = 100000000;
    auto str = my::to_string(x);
    BOOST_ASSERT(str == "100,000,000");

    my::append(str) (" in hexadecimal is ", ~strf::hex(x));
    BOOST_ASSERT(str == "100,000,000 in hexadecimal is 0x5f5'e100");

    char buff[500];
    my::write(buff)(x, " in hexadecimal is ", ~strf::hex(x));
    BOOST_ASSERT(str == buff);

    // Overriding numpunct_c<16> back to default:
    str = my::to_string
        .facets(strf::no_grouping<16>())
        (x, " in hexadecimal is ", ~strf::hex(x));
    BOOST_ASSERT(str == "100,000,000 in hexadecimal is 0x5f5e100");
}
//]

int main()
{
    sample();
    format_functions();
    format_functions_2();
    leading_expression_exception();
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
    width_as_len();
    width_as_u32len();
    width_func();
    using_my_customizations();
    return 0;
}
