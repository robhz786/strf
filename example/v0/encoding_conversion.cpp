#include <boost/assert.hpp>
#include <boost/stringify.hpp>

#include <iostream>

void input_ouput_different_char_types()
{
    //[input_output_different_char_types
    namespace strf = boost::stringify::v0;

    auto str   = strf::make_string    ("aaa-", u"bbb-", U"ccc-", L"ddd");
    auto str16 = strf::make_u16string ("aaa-", u"bbb-", U"ccc-", L"ddd");
    auto str32 = strf::make_u32string ("aaa-", u"bbb-", U"ccc-", L"ddd");
    auto wstr  = strf::make_wstring   ("aaa-", u"bbb-", U"ccc-", L"ddd");

    BOOST_ASSERT(str.value()   ==  "aaa-bbb-ccc-ddd");
    BOOST_ASSERT(str16.value() == u"aaa-bbb-ccc-ddd");
    BOOST_ASSERT(str32.value() == U"aaa-bbb-ccc-ddd");
    BOOST_ASSERT(wstr.value()  == L"aaa-bbb-ccc-ddd");
    //]
}


void mutf8 ()
{

    //[ mutf8_sample
    namespace strf = boost::stringify::v0;

    // from UTF-16 to  Modified UTF-8 (MTF-8)
    std::u16string str_utf16 {u"---\0---", 7};
    auto str_mutf8 = strf::make_string.facets(strf::mutf8()) (str_utf16);

    BOOST_ASSERT(str_mutf8.value() == "---\xC0\x80---");

    // from Modified UTF-8 (MTF-8) back to UTF-16
    auto str_utf16_2 = strf::make_u16string.facets(strf::mutf8()) (str_mutf8.value());
    BOOST_ASSERT(str_utf16 == str_utf16_2.value());
    //]
}

void arg()
{
    //[ arg_encoding
    namespace strf = boost::stringify::v0;

    auto str_utf8 = strf::make_string
        ( strf::fmt("--\xA4--").encoding(strf::iso_8859_1())
        , strf::fmt("--\xA4--").encoding(strf::iso_8859_15()));

    BOOST_ASSERT(str_utf8.value() == u8"--\u00A4----\u20AC--");
    //]
}


void arg_abbreviated()
{
    //[ arg_encoding_abbreviated
    namespace strf = boost::stringify::v0;

    auto str_utf8 = strf::make_string
        ( strf::iso_8859_1("--\xA4--")
        , strf::iso_8859_15("--\xA4--") );

    BOOST_ASSERT(str_utf8.value() == u8"--\u00A4----\u20AC--");
    //]
}

void not_sanitized_inputs()
{
    //[ not_sanitized_inputs
    namespace strf = boost::stringify::v0;

    auto str_utf8 = strf::make_string
        ( "--\xbf\xbf--"                  // non-conformant UTF-8
        , strf::ascii("--\x80--")         // non-conformant ASCII
        , strf::iso_8859_1("--\x80--") ); // non-conformant ISO 8859-1

    // The result is an invalid UTF-8 output.
    // Only the string in ISO 8859-1 is sanitized:
    BOOST_ASSERT(str_utf8.value() == std::string{"--\xbf\xbf----\x80----"} + u8"\uFFFD--");
    //]
}

void sanitized_inputs()
{
    //[ sanitized_inputs
    namespace strf = boost::stringify::v0;

    auto str_utf8 = strf::make_string
        ( strf::sani("--\xbf\xbf--")             // non-conformant UTF-8
        , strf::ascii("--\x80--").sani()         // non-conformant ASCII
        , strf::iso_8859_1("--\x80--").sani() ); // non-conformant ISO 8859-1

    BOOST_ASSERT(str_utf8.value() == u8"--\uFFFD----\uFFFD----\uFFFD--");
    //]
}

void error_signal_char()
{
    //[ error_signal_char
    namespace strf = boost::stringify::v0;

    strf::encoding_error enc_err{U'!'};
    auto str = strf::make_string .facets(enc_err) (strf::sani("--\x99--"));

    BOOST_ASSERT(str.value() == "--!--");
    //]
}

void error_signal_skip()
{
    //[ error_signal_skip
    namespace strf = boost::stringify::v0;

    strf::encoding_error enc_err{};
    auto str = strf::make_string .facets(enc_err) (strf::sani("--\x99--"));

    BOOST_ASSERT(str.value() == "----");
    //]
}


void error_signal_code()
{
    //[error_signal_code
    namespace strf = boost::stringify::v0;

    std::error_code ec{std::make_error_code(std::errc::illegal_byte_sequence)};
    strf::encoding_error enc_err{ec};

    auto str = strf::make_string .facets(enc_err) (strf::sani("--\x99--"));

    BOOST_ASSERT(!str && str.error() == ec);
    //]
}


//[ error_signal_throw
void thrower_func()
{
    throw std::invalid_argument("encoding error");
}

void sample()
{
    std::string what;

    try
    {
        namespace strf = boost::stringify::v0;
        strf::encoding_error enc_err{thrower_func};
        (void) strf::make_string .facets(enc_err) (strf::sani("--\x99--"));
    }
    catch(std::invalid_argument& e)
    {
        what = e.what();
    }

    BOOST_ASSERT(what == "encoding error");
}
//]

void keep_surrogates ()
{
    //[ keep_surrogates
    namespace strf = boost::stringify::v0;

    std::u16string input_utf16 {u"-----"};
    input_utf16[1] = 0xD800; // a surrogate character alone


    auto str1 = strf::make_string
        .facets(strf::keep_surrogates{false})
        (input_utf16);

    auto str2 = strf::make_string
        .facets(strf::keep_surrogates{true})
        (input_utf16);


    BOOST_ASSERT(str1.value() == u8"-\uFFFD---");
    BOOST_ASSERT(str2.value() ==   "-\xED\xA0\x80---");

    // now back to UTF-16
    auto utf16_no_surr = strf::make_u16string
        .facets(strf::keep_surrogates{false})
        (str2.value());

    auto utf16_with_surr = strf::make_u16string
        .facets(strf::keep_surrogates{true})
        (str2.value());

    BOOST_ASSERT(utf16_no_surr.value() == u"-\uFFFD---");
    BOOST_ASSERT(utf16_with_surr.value()[1] == 0xD800);
    //]

}


int main()
{
    input_ouput_different_char_types();
    mutf8 ();
    arg();
    arg_abbreviated();
    not_sanitized_inputs();
    sanitized_inputs();
    error_signal_char();
    error_signal_code();
    sample();
    keep_surrogates();

    return 0;
}

