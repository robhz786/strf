#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"

namespace strf = boost::stringify::v0;

template <typename CharT>
std::basic_string<CharT> str_0_to_9(strf::encoding<CharT>)
{
    return {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };
}
template <typename CharT>
std::basic_string<CharT> str_abcde(strf::encoding<CharT>)
{
    return {'a', 'b', 'c', 'd', 'e'};
}

template <typename CharT>
std::basic_string<CharT> str_0_to_9_failed_last(strf::encoding<CharT>)
{
    CharT invalid_char = std::is_same<CharT, char>::value ? (CharT)'\x81' : (CharT)0xDFFF;
    return { '0', '1', '2', '3', '4', '5', '6', '7', '8', invalid_char };
}

template <typename CharT>
std::basic_string<CharT> str_abcde_failed_first(strf::encoding<CharT>)
{
    CharT invalid_char = std::is_same<CharT, char>::value ? (CharT)'\x81' : (CharT)0xDFFF;
    return { invalid_char, 'b', 'c', 'd', 'e' };
}

template <typename CharT>
strf::encoding_error replacement_facet(strf::encoding<CharT> enc)
{
    if ( enc == strf::ascii() || enc == strf::iso_8859_1()
      || enc == strf::iso_8859_15() || enc == strf::windows_1252()
    )
    {
        return { U'?' };
    }
    return {U'\U00101000'};
}

template <typename CharIn, typename CharOut>
void test_in_buffered_output(strf::encoding<CharIn> ein, strf::encoding<CharOut> eout)
{
    std::error_code errcode = std::make_error_code(std::errc::illegal_byte_sequence);

    auto repstr = strf::to_basic_string<CharOut>
        (replacement_facet(eout).on_error().get_char())
        .value();

    const auto repstr_x12 = strf::to_basic_string<CharOut>
        ( repstr, repstr, repstr, repstr, repstr, repstr
        , repstr, repstr, repstr, repstr, repstr, repstr )
        .value();

    // when an invalid char is replaced
    // just before the buffer capacity is reached
    BUFFERED_TEST(10, str_0_to_9(eout).substr(0, 9) + repstr + str_abcde(eout))
        .facets(eout, strf::keep_surrogates{false}, replacement_facet(eout))
        (strf::sani(str_0_to_9_failed_last(ein) + str_abcde(ein)).encoding(ein));

    // when an invalid char is replaced
    // just after the buffer capacity was reached
    BUFFERED_TEST(10, str_0_to_9(eout) + repstr + str_abcde(eout).substr(1, 5))
        .facets(eout, strf::keep_surrogates{false}, replacement_facet(eout))
        (strf::sani(str_0_to_9(ein) + str_abcde_failed_first(ein)).encoding(ein));

    // when an invalid char is removed
    // just before the buffer capacity is reached
    BUFFERED_TEST_RF(10, str_0_to_9(eout).substr(0, 9) + str_abcde(eout), 1.5)
        .facets(eout, strf::keep_surrogates{false}, strf::encoding_error())
        (strf::sani(str_0_to_9_failed_last(ein) + str_abcde(ein)).encoding(ein));

    // when an invalid char is removed
    // just after the buffer capacity was reached
    BUFFERED_TEST_RF(10, str_0_to_9(eout) + str_abcde(eout).substr(1, 5), 1.5)
        .facets(eout, strf::keep_surrogates{false}, strf::encoding_error())
        (strf::sani(str_0_to_9(ein) + str_abcde_failed_first(ein)).encoding(ein));

    // when an invalid char causes an error code
    // just before the buffer capacity is reached
    BUFFERED_TEST_ERR(10, str_0_to_9(eout).substr(0, 9), errcode)
        .facets(eout, strf::keep_surrogates{false}, strf::encoding_error(errcode))
        (strf::sani(str_0_to_9_failed_last(ein) + str_abcde(ein)).encoding(ein));

    // when an invalid char causes an error code
    // just after the buffer capacity was reached
    BUFFERED_TEST_ERR(10, str_0_to_9(eout), errcode)
        .facets(eout, strf::keep_surrogates{false}, strf::encoding_error(errcode))
        (strf::sani(str_0_to_9(ein) + str_abcde_failed_first(ein)).encoding(ein));


    //TODO:  when compilable, replace char32_t by CharIn and use ein.
    using CharInT = char32_t;

    auto expected = std::basic_string<CharOut>(15, CharOut('a'));
    BUFFERED_TEST(10, expected)
        .facets(eout)
        (strf::multi(CharInT('a'), 15));

    expected.clear();
    expected.append(10, CharOut('a'));
    expected.append(5, CharOut('b'));
    BUFFERED_TEST(10, expected)
        .facets(eout, strf::keep_surrogates{false}, replacement_facet(eout))
        (strf::multi(CharInT('a'), 10), strf::multi(CharInT('b'), 5));

    //if(std::is_same(char32_t, CharOut)::value) // TODO remove this condition as soon as .sani is supporter
    //{
    //    BUFFERED_TEST(10, repstr_x12 + std::basic_string<CharOut>(5, CharOut('b')))
    //        .facets(eout, strf::keep_surrogates{ false }, replacement_facet(eout))
    //        (strf::multi(invalid_char, 12)/*.sani() TODO */, strf::multi(CharInT('b'), 5));
    //}
}


void test_utf8_variant(strf::encoding<char> enc)
{
    std::error_code errcode = std::make_error_code(std::errc::illegal_byte_sequence);

    const std::u32string u32sample
        = { U'a', ' ', 0x80, ' ', 0x800, ' ', 0x10000, /*' ', 0xD800, ' ', 0xDC00,*/ ' ', 0x10FFFF, ' '};

    const std::u16string u16sample
        = { U'a', ' ', 0x80, ' ', 0x800, ' ',  0xD800, 0xDC00, ' ', 0xDBFF, 0xDFFF, ' '};

    const std::u32string u32_invalid_chars
        = { ' ', 0xD800, ' ', 0xDBFF, ' ', 0xDC00, ' ', 0xDFFF, ' ', 0x110000, ' ' };

    const std::u32string u16_invalid_chars
        = { ' ', 0xD800, ' ', 0xDBFF, ' ', 0xDC00, ' ', 0xDFFF, ' '};

    TEST(u8"a \u0080 \u0800 \U00010000 \U0010FFFF ").facets(enc) (u32sample);
    TEST(u8"a \u0080 \u0800 \U00010000 \U0010FFFF ").facets(enc) (u16sample);

    TEST(" \xED\xA0\x80 \xED\xAF\xBF \xED\xB0\x80 \xED\xBF\xBF \xEF\xBF\xBD ")
        .facets(enc, strf::keep_surrogates{ true })
        (u32_invalid_chars);
    TEST(" \xED\xA0\x80 \xED\xAF\xBF \xED\xB0\x80 \xED\xBF\xBF ")
        .facets(enc, strf::keep_surrogates{ true })
        (u16_invalid_chars);

    TEST(u8" \uFFFD \uFFFD \uFFFD \uFFFD \uFFFD ")
        .facets(enc, strf::keep_surrogates{ false })
        (u32_invalid_chars);
    TEST(u8" \uFFFD \uFFFD \uFFFD \uFFFD ")
        .facets(enc, strf::keep_surrogates{ false })
        (u16_invalid_chars);

    TEST(u8" * * * * * ")
        .facets(enc, strf::keep_surrogates{ false }, strf::encoding_error{ U'*' })
        (u32_invalid_chars);
    TEST(u8" * * * * ")
        .facets(enc, strf::keep_surrogates{ false }, strf::encoding_error{ U'*' })
        (u16_invalid_chars);
    TEST(u8"      ")
        .facets(enc, strf::keep_surrogates{ false }, strf::encoding_error{})
        (u32_invalid_chars);
    TEST(u8"     ")
        .facets(enc, strf::keep_surrogates{ false }, strf::encoding_error{})
        (u16_invalid_chars);

    TEST_ERR(" \xED\xA0\x80 \xED\xAF\xBF \xED\xB0\x80 \xED\xBF\xBF ", errcode)
        .facets(enc, strf::keep_surrogates{ true }, strf::encoding_error(errcode))
        (u32_invalid_chars);

    // from UTF-8
    TEST(u32sample).facets(enc) (u8"a \u0080 \u0800 \U00010000 \U0010FFFF ");
    TEST(u16sample).facets(enc) (u8"a \u0080 \u0800 \U00010000 \U0010FFFF ");
    TEST(u16_invalid_chars)
        .facets(enc, strf::keep_surrogates{ true })
        (" \xED\xA0\x80 \xED\xAF\xBF \xED\xB0\x80 \xED\xBF\xBF ");

    // sanitization

    TEST(" * * * - * * * - * - * * * * ")
        .facets(enc, strf::keep_surrogates(false), strf::encoding_error(U'*'))
        ( strf::sani
           ( " \xDF \xEF\x80 \xF3\x80\x80 " //leading byte with not enough continuation bytes
             "- \x80 \x80\x80 \xBF\xBF\x80 " //continuation bytes not preceeded by a leading byte
             "- \xF5\xBF\xBF\xBF " // codepoint too big
             "- \xED\xA0\x80 \xED\xAF\xBF \xED\xB0\x80 \xED\xBF\xBF ")); // surrogates

    TEST(" * * * - * * * - * - \xED\xA0\x80 \xED\xAF\xBF \xED\xB0\x80 \xED\xBF\xBF ")
        .facets(enc, strf::keep_surrogates(true), strf::encoding_error(U'*'))
        ( strf::sani
            ( " \xDF \xEF\x80 \xF3\x80\x80 " //leading byte with not enough continuation bytes
              "- \x80 \x80\x80 \xBF\xBF\x80 " //continuation bytes not preceeded by a leading byte
              "- \xF5\xBF\xBF\xBF " // codepoint too big
              "- \xED\xA0\x80 \xED\xAF\xBF \xED\xB0\x80 \xED\xBF\xBF ")); // surrogates

    //codepoint too big
    BUFFERED_TEST(10, u8"012345678\uFFFD abc") (strf::sani("012345678\xF5\xBF\xBF\xBF\xBF\xBF\xBF abc").encoding(enc));
    BUFFERED_TEST_RF(10,  u"012345678\uFFFD abc", 1.25) (strf::sani("012345678\xF5\xBF\xBF\xBF\xBF\xBF\xBF abc").encoding(enc));
    BUFFERED_TEST(10,  U"012345678\uFFFD abc") (strf::sani("012345678\xF5\xBF\xBF\xBF\xBF\xBF\xBF abc").encoding(enc));

    //continuation bytes not preceeded by a leading byte
    BUFFERED_TEST(10, u8"012345678\uFFFD abc") (strf::sani("012345678\xBF\xBF\xBF\xBF\xBF\xBF\xBF abc").encoding(enc));
    BUFFERED_TEST_RF(10,  u"012345678\uFFFD abc", 1.1) (strf::sani("012345678\xBF\xBF\xBF\xBF\xBF\xBF\xBF abc").encoding(enc));
    BUFFERED_TEST(10,  U"012345678\uFFFD abc") (strf::sani("012345678\xBF\xBF\xBF\xBF\xBF\xBF\xBF abc").encoding(enc));

    BUFFERED_TEST(10, u8"012345678\uFFFD abc")
        .facets(strf::keep_surrogates(false))
        (strf::sani("012345678\xED\xA0\x80\xBF\xBF\xBF\xBF\xBF\xBF\xBF abc").encoding(enc));
    BUFFERED_TEST_RF(10, u"012345678\uFFFD abc", 1.25)
        .facets(strf::keep_surrogates(false))
        (strf::sani("012345678\xED\xA0\x80\xBF\xBF\xBF\xBF\xBF\xBF\xBF abc").encoding(enc));
    BUFFERED_TEST(10, U"012345678\uFFFD abc")
        .facets(strf::keep_surrogates(false))
        (strf::sani("012345678\xED\xA0\x80\xBF\xBF\xBF\xBF\xBF\xBF\xBF abc").encoding(enc));
}


int main()
{
    // ASCII

    // ascii sanitization
    TEST("abc\x80 abc\x7f?").facets(strf::ascii()) ("abc\x80 ", strf::sani("abc\x7F\x80"));
    // from ascii to iso_8859_1
    TEST("abc?").facets(strf::iso_8859_1()) (strf::ascii("abc\xEE").sani());
    // from ascii to utf-8
    TEST(u8"abc\uFFFD").facets(strf::utf8()) (strf::ascii("abc\xEE").sani());
    // utf8 to ASCII
    TEST("abc ?? def").facets(strf::ascii()) (strf::utf8(u8"abc \u1000\u1001 def"));

    // ISO 8859-1

    // ISO 8859-1 sanitization
    TEST("\x7F ?? \xA0").facets(strf::iso_8859_1()) (strf::sani("\x7F \x80\x9F \xA0"));
    // ISO 8859-1 to utf8
    TEST(u8"\x7F \uFFFD\uFFFD \u00A0") (strf::iso_8859_1("\x7F \x80\x9F \xA0"));
    // ISO 8859-1 to ASCII
    TEST("\x7F???").facets(strf::ascii()) (strf::iso_8859_1("\x7F\x80\x9F\x80"));
    // ISO 8859-1 to 8859-15
    TEST("abc\x7F\xA3\xBF\xFF????????")
        .facets(strf::iso_8859_15())
        (strf::iso_8859_1("abc\x7F\xA3\xBF\xFF\xA4\xA6\xA8\xB4\xB8\xBC\xBD\xBE"));
    // sani ISO 8859-1 to 8859-15
    TEST("abc\x7F\xA3\xBF\xFF??").facets(strf::iso_8859_15())
        (strf::iso_8859_1("abc\x7F\xA3\xBF\xFF\x80\x9F").sani());
    // sani ISO 8859-1 to windows-1252
    TEST("abc\x7F\xA3\xBF\xFF??").facets(strf::windows_1252())
        (strf::iso_8859_1("abc\x7F\xA3\xBF\xFF\x80\x9F").sani());

    // ISO 8859-15

    // ISO 8859-15 sanitization
    TEST("\x7F ?? \xA0").facets(strf::iso_8859_15())
        (strf::sani("\x7F \x80\x9F \xA0"));
    // ISO 8859-15 to utf8
    TEST(u8"\x7F \uFFFD\uFFFD \u20AC\u0160\u0161\u017D\u017E\u0152\u0153\u0178")
        (strf::iso_8859_15("\x7F \x80\x9F \xA4\xA6\xA8\xB4\xB8\xBC\xBD\xBE"));
    // ISO 8859-15 to utf16
    TEST(u"\u007F\uFFFD\uFFFD \u20AC\u0160\u0161\u017D\u017E\u0152\u0153\u0178")
        (strf::iso_8859_15("\x7F\x80\x9F \xA4\xA6\xA8\xB4\xB8\xBC\xBD\xBE"));
    // ISO 8859-15 to ISO 8859-1
    TEST(u8"\x7F \uFFFD\uFFFD \u20AC\u0160\u0161\u017D\u017E\u0152\u0153\u0178")
        (strf::iso_8859_15("\x7F \x80\x9F \xA4\xA6\xA8\xB4\xB8\xBC\xBD\xBE"));

    const char* str_80_to_9F =
        "\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8A\x8B\x8C\x8D\x8E\x8F"
        "\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9A\x9B\x9C\x9D\x9E\x9F";

    // WINDOWS 1252
    // WINDOWS 1252 sanitization
    TEST("\x7F ????? \xA0").facets(strf::windows_1252())
        (strf::sani("\x7F \x81\x8D\x8F\x90\x9D \xA0"));
    // WINDOWS 1252 to utf8
    TEST(u8"abc\u00b0\x7F \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD \u00A0")
        (strf::windows_1252("abc\xb0\x7F \x81\x8D\x8F\x90\x9D \xA0"));
    TEST(u8"\u20AC\uFFFD\u201A\u0192\u201E\u2026\u2020\u2021\u02C6\u2030\u0160"
         u8"\u2039\u0152\uFFFD\u017D\uFFFD\uFFFD\u2018\u2019\u201C\u201D\u2022"
         u8"\u2013\u2014\u02DC\u2122\u0161\u203A\u0153\uFFFD\u017E\u0178")
        (strf::windows_1252(str_80_to_9F));


    // WINDOWS 1252 to ISO 8859-1
    TEST("????????????????????????????????")
        .facets(strf::iso_8859_1())
        (strf::windows_1252(str_80_to_9F));
    TEST("abc\xA0\xFF").facets(strf::iso_8859_1())
        (strf::windows_1252("abc\xA0\xFF"));

    // UTF-8 and MUTF-8

    test_utf8_variant(strf::utf8());
    test_utf8_variant(strf::mutf8());
    // MUTF-8 to UTF-8
    TEST(std::string("--\0--", 5)) ( strf::mutf8("--\xC0\x80--") );
    // UTF-8 to MUTF-8
    TEST("--\xC0\x80--").facets(strf::mutf8())(strf::utf8(std::string("--\0--", 5)));
    TEST("--\xC0\x80--").facets(strf::mutf8())(strf::sani(std::string("--\0--", 5)));
    // UTF-16 to MUTF-8
    TEST("--\xC0\x80--").facets(strf::mutf8())(std::u16string(u"--\0--", 5));

    // UTF-16
    TEST_RF(u"--\uFFFD--", 1.2).facets(strf::keep_surrogates(false))("--\xED\xA0\x80--");
    {
        std::u16string expected{ '-', '-', 0xD800, '-', '-' };
        TEST(expected).facets(strf::keep_surrogates(true))("--\xED\xA0\x80--");
    }
    BUFFERED_TEST_RF(10, u"012345678\uFFFD\uFFFD", 1.2)
        .facets(strf::keep_surrogates(false))
        ("012345678", strf::sani(std::u16string(2, char16_t(0xD800))));
    BUFFERED_TEST_RF(10, u8"012345678\uFFFD\uFFFD", 1.2)
        .facets(strf::keep_surrogates(false))
        ("012345678", std::u16string(2, char16_t(0xD800)));
    BUFFERED_TEST_RF(10, U"012345678\uFFFD\uFFFD", 1.2)
        .facets(strf::keep_surrogates(false))
        ("012345678", std::u16string(2, char16_t(0xD800)));

    // TODO: more UTF-16 tests

    // convertion in buffered output

    test_in_buffered_output(strf::ascii(), strf::ascii());
    test_in_buffered_output(strf::ascii(), strf::iso_8859_1());
    test_in_buffered_output(strf::ascii(), strf::iso_8859_15());
    test_in_buffered_output(strf::ascii(), strf::windows_1252());
    test_in_buffered_output(strf::ascii(), strf::utf8());
    test_in_buffered_output(strf::ascii(), strf::mutf8());
    test_in_buffered_output(strf::ascii(), strf::utf16<char16_t>());
    test_in_buffered_output(strf::ascii(), strf::utf32<char32_t>());
    test_in_buffered_output(strf::ascii(), strf::utfw());

    test_in_buffered_output(strf::iso_8859_1(), strf::ascii());
    test_in_buffered_output(strf::iso_8859_1(), strf::iso_8859_1());
    test_in_buffered_output(strf::iso_8859_1(), strf::iso_8859_15());
    test_in_buffered_output(strf::iso_8859_1(), strf::windows_1252());
    test_in_buffered_output(strf::iso_8859_1(), strf::utf8());
    test_in_buffered_output(strf::iso_8859_1(), strf::mutf8());
    test_in_buffered_output(strf::iso_8859_1(), strf::utf16<char16_t>());
    test_in_buffered_output(strf::iso_8859_1(), strf::utf32<char32_t>());
    test_in_buffered_output(strf::iso_8859_1(), strf::utfw());

    test_in_buffered_output(strf::iso_8859_15(), strf::ascii());
    test_in_buffered_output(strf::iso_8859_15(), strf::iso_8859_1());
    test_in_buffered_output(strf::iso_8859_15(), strf::iso_8859_15());
    test_in_buffered_output(strf::iso_8859_15(), strf::windows_1252());
    test_in_buffered_output(strf::iso_8859_15(), strf::utf8());
    test_in_buffered_output(strf::iso_8859_15(), strf::mutf8());
    test_in_buffered_output(strf::iso_8859_15(), strf::utf16<char16_t>());
    test_in_buffered_output(strf::iso_8859_15(), strf::utf32<char32_t>());
    test_in_buffered_output(strf::iso_8859_15(), strf::utfw());

    test_in_buffered_output(strf::windows_1252(), strf::ascii());
    test_in_buffered_output(strf::windows_1252(), strf::iso_8859_1());
    test_in_buffered_output(strf::windows_1252(), strf::iso_8859_15());
    test_in_buffered_output(strf::windows_1252(), strf::windows_1252());
    test_in_buffered_output(strf::windows_1252(), strf::utf8());
    test_in_buffered_output(strf::windows_1252(), strf::mutf8());
    test_in_buffered_output(strf::windows_1252(), strf::utf16<char16_t>());
    test_in_buffered_output(strf::windows_1252(), strf::utf32<char32_t>());
    test_in_buffered_output(strf::windows_1252(), strf::utfw());

    test_in_buffered_output(strf::utf8(), strf::ascii());
    test_in_buffered_output(strf::utf8(), strf::iso_8859_1());
    test_in_buffered_output(strf::utf8(), strf::iso_8859_15());
    test_in_buffered_output(strf::utf8(), strf::windows_1252());
    test_in_buffered_output(strf::utf8(), strf::utf8());
    test_in_buffered_output(strf::utf8(), strf::mutf8());
    test_in_buffered_output(strf::utf8(), strf::utf16<char16_t>());
    test_in_buffered_output(strf::utf8(), strf::utf32<char32_t>());
    test_in_buffered_output(strf::utf8(), strf::utfw());

    test_in_buffered_output(strf::mutf8(), strf::ascii());
    test_in_buffered_output(strf::mutf8(), strf::iso_8859_1());
    test_in_buffered_output(strf::mutf8(), strf::iso_8859_15());
    test_in_buffered_output(strf::mutf8(), strf::windows_1252());
    test_in_buffered_output(strf::mutf8(), strf::utf8());
    test_in_buffered_output(strf::mutf8(), strf::mutf8());
    test_in_buffered_output(strf::mutf8(), strf::utf16<char16_t>());
    test_in_buffered_output(strf::mutf8(), strf::utf32<char32_t>());
    test_in_buffered_output(strf::mutf8(), strf::utfw());

    test_in_buffered_output(strf::utf16<char16_t>(), strf::ascii());
    test_in_buffered_output(strf::utf16<char16_t>(), strf::iso_8859_1());
    test_in_buffered_output(strf::utf16<char16_t>(), strf::iso_8859_15());
    test_in_buffered_output(strf::utf16<char16_t>(), strf::windows_1252());
    test_in_buffered_output(strf::utf16<char16_t>(), strf::utf8());
    test_in_buffered_output(strf::utf16<char16_t>(), strf::mutf8());
    test_in_buffered_output(strf::utf16<char16_t>(), strf::utf16<char16_t>());
    test_in_buffered_output(strf::utf16<char16_t>(), strf::utf32<char32_t>());
    test_in_buffered_output(strf::utf16<char16_t>(), strf::utfw());

    test_in_buffered_output(strf::utf32<char32_t>(), strf::ascii());
    test_in_buffered_output(strf::utf32<char32_t>(), strf::iso_8859_1());
    test_in_buffered_output(strf::utf32<char32_t>(), strf::iso_8859_15());
    test_in_buffered_output(strf::utf32<char32_t>(), strf::windows_1252());
    test_in_buffered_output(strf::utf32<char32_t>(), strf::utf8());
    test_in_buffered_output(strf::utf32<char32_t>(), strf::mutf8());
    test_in_buffered_output(strf::utf32<char32_t>(), strf::utf16<char16_t>());
    test_in_buffered_output(strf::utf32<char32_t>(), strf::utf32<char32_t>());
    test_in_buffered_output(strf::utf32<char32_t>(), strf::utfw());

    test_in_buffered_output(strf::utfw(), strf::ascii());
    test_in_buffered_output(strf::utfw(), strf::iso_8859_1());
    test_in_buffered_output(strf::utfw(), strf::iso_8859_15());
    test_in_buffered_output(strf::utfw(), strf::windows_1252());
    test_in_buffered_output(strf::utfw(), strf::utf8());
    test_in_buffered_output(strf::utfw(), strf::mutf8());
    test_in_buffered_output(strf::utfw(), strf::utf16<char16_t>());
    test_in_buffered_output(strf::utfw(), strf::utf32<char32_t>());
    test_in_buffered_output(strf::utfw(), strf::utfw());

    return report_errors() || boost::report_errors();
}
