//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "lightweight_test_label.hpp"
#include <strf.hpp>
#include <vector>

template <typename CharT>
std::basic_string<CharT> repeat
    ( std::size_t count
    , std::basic_string<CharT> str )
{
    std::basic_string<CharT> x;
    x.reserve(count * str.size());
    while (count--)
    {
        x.append(str);
    }
    return x;
}

template <typename CharT>
inline std::basic_string<CharT> repeat(std::size_t count, const CharT* str)
{
    return repeat<CharT>(count, std::basic_string<CharT>{str});
}

template <typename CharT>
void test_fill
    ( strf::encoding<CharT> enc
    , char32_t fill_char
    , std::basic_string<CharT> encoded_char
    , strf::surrogate_policy allow_surr = strf::surrogate_policy::strict )
{
    BOOST_TEST_LABEL << enc.name() << ", test_fill_char: U+"
                     << std::hex << (unsigned)fill_char
                     << std::dec;

    {
        std::int16_t count = 10;
        auto result = strf::to_basic_string<CharT>
            .with(enc, strf::encoding_error::replace, allow_surr)
            (strf::right(CharT('x'), 11, fill_char));

        auto expected = repeat(count, encoded_char);
        expected.push_back(CharT('x'));

        BOOST_TEST(result == expected);
    }
    {
        std::int16_t count = 200;
        auto result = strf::to_basic_string<CharT>
            .with(enc, strf::encoding_error::replace, allow_surr)
            (strf::right(CharT('x'), count + 1, fill_char));

        auto expected = repeat(count, encoded_char);
        expected.push_back(CharT('x'));

        BOOST_TEST(result == expected);
    }
}

template <typename CharT>
inline void test_fill
    ( strf::encoding<CharT> enc
    , char32_t fill_char
    , const CharT* encoded_char
    , strf::surrogate_policy allow_surr = strf::surrogate_policy::strict )
{
    return test_fill
        ( enc
        , fill_char
        , std::basic_string<CharT>{encoded_char}
        , allow_surr );
}


template <typename CharT>
void test_invalid_fill_stop
    ( strf::encoding<CharT> enc
    , char32_t fill_char
    , strf::surrogate_policy allow_surr = strf::surrogate_policy::strict )
{
    BOOST_TEST_LABEL << "encoding: " << enc.name()
                     << "; test_fill_char: \\u'"
                     << std::hex << (unsigned)fill_char << '\''
                     << std::dec;

    {
        auto facets = strf::pack(enc, strf::encoding_error::stop, allow_surr);
        BOOST_TEST_THROWS( (strf::to_string.with(facets)(strf::right(0, 10, fill_char)))
                         , strf::encoding_failure );
    }
}

template <typename CharT>
void test_invalid_fill_ignore
    ( strf::encoding<CharT> enc
    , char32_t fill_char
    , strf::surrogate_policy allow_surr = strf::surrogate_policy::strict )
{
    BOOST_TEST_LABEL << "test_fill_char: \\u'"
                     << std::hex << (unsigned)fill_char << '\''
                     << std::dec;

    std::basic_string<CharT> expected(5, CharT('-'));
    expected.push_back(CharT('x'));
    for(int i = 0; i < 5; ++i)
    {
        expected.push_back(CharT('+'));
    }

    {
        auto result = strf::to_basic_string<CharT>
            .with(enc, strf::encoding_error::ignore, allow_surr)
            ( strf::multi(CharT('-'), 5)
            , strf::right(CharT('x'), 11, fill_char)
            , strf::multi(CharT('+'), 5) );
        BOOST_TEST(result == expected);
    }
}


int main()
{
    {
        // UTF-8

        test_fill(strf::utf8<char>(), 0x7F, "\x7F");
        test_fill(strf::utf8<char>(), 0x80, "\xC2\x80");
        test_fill(strf::utf8<char>(), 0x800, "\xE0\xA0\x80");
        test_fill(strf::utf8<char>(), 0xD800, "\xED\xA0\x80", strf::surrogate_policy::lax);
        test_fill(strf::utf8<char>(), 0xDBFF, "\xED\xAF\xBF", strf::surrogate_policy::lax);
        test_fill(strf::utf8<char>(), 0xDC00, "\xED\xB0\x80", strf::surrogate_policy::lax);
        test_fill(strf::utf8<char>(), 0xDFFF, "\xED\xBF\xBF", strf::surrogate_policy::lax);
        test_fill(strf::utf8<char>(), 0xFFFF, "\xEF\xBF\xBF");
        test_fill(strf::utf8<char>(), 0x10000, "\xF0\x90\x80\x80");
        test_fill(strf::utf8<char>(), 0x10FFFF, "\xF4\x8F\xBF\xBF");


        test_fill(strf::utf8<char>(), 0xD800, "\xEF\xBF\xBD");
        test_fill(strf::utf8<char>(), 0xDBFF, "\xEF\xBF\xBD");
        test_fill(strf::utf8<char>(), 0xDC00, "\xEF\xBF\xBD");
        test_fill(strf::utf8<char>(), 0xDFFF, "\xEF\xBF\xBD");
        test_fill(strf::utf8<char>(), 0x110000, "\xEF\xBF\xBD");
        test_invalid_fill_stop(strf::utf8<char>(), 0x110000);
        test_invalid_fill_ignore(strf::utf8<char>(), 0x110000);
    }

    {
        // UTF-16;
        test_fill(strf::utf16<char16_t>(), U'a', u"a");
        test_fill(strf::utf16<char16_t>(), 0xD800, {0xD800}, strf::surrogate_policy::lax);
        test_fill(strf::utf16<char16_t>(), 0xDBFF, {0xDBFF}, strf::surrogate_policy::lax);
        test_fill(strf::utf16<char16_t>(), 0xDC00, {0xDC00}, strf::surrogate_policy::lax);
        test_fill(strf::utf16<char16_t>(), 0xDFFF, {0xDFFF}, strf::surrogate_policy::lax);
        test_fill(strf::utf16<char16_t>(), 0x10000,  u"\U00010000");
        test_fill(strf::utf16<char16_t>(), 0x10FFFF, u"\U0010FFFF");

        test_fill(strf::utf16<char16_t>(), 0xD800, u"\uFFFD");
        test_fill(strf::utf16<char16_t>(), 0xDBFF, u"\uFFFD");
        test_fill(strf::utf16<char16_t>(), 0xDC00, u"\uFFFD");
        test_fill(strf::utf16<char16_t>(), 0xDFFF, u"\uFFFD");
        test_fill(strf::utf16<char16_t>(), 0x110000, u"\uFFFD");
        test_invalid_fill_stop(strf::utf16<char16_t>(), 0x110000);
        test_invalid_fill_ignore(strf::utf16<char16_t>(), 0x110000);
    }

    {
        // UTF-32;
        test_fill(strf::utf32<char32_t>(), U'a', U"a");
        test_fill(strf::utf32<char32_t>(), 0xD800, {0xD800}, strf::surrogate_policy::lax);
        test_fill(strf::utf32<char32_t>(), 0xDBFF, {0xDBFF}, strf::surrogate_policy::lax);
        test_fill(strf::utf32<char32_t>(), 0xDC00, {0xDC00}, strf::surrogate_policy::lax);
        test_fill(strf::utf32<char32_t>(), 0xDFFF, {0xDFFF}, strf::surrogate_policy::lax);
        test_fill(strf::utf32<char32_t>(), 0x10000,  U"\U00010000");
        test_fill(strf::utf32<char32_t>(), 0x10FFFF, U"\U0010FFFF");

        test_fill(strf::utf32<char32_t>(), 0xD800, U"\uFFFD");
        test_fill(strf::utf32<char32_t>(), 0xDBFF, U"\uFFFD");
        test_fill(strf::utf32<char32_t>(), 0xDC00, U"\uFFFD");
        test_fill(strf::utf32<char32_t>(), 0xDFFF, U"\uFFFD");
        test_fill(strf::utf32<char32_t>(), 0x110000, U"\uFFFD");
        test_invalid_fill_stop(strf::utf32<char32_t>(), 0x110000);
        test_invalid_fill_ignore(strf::utf32<char32_t>(), 0x110000);
    }

    {
        // single byte encodings
        test_fill(strf::windows_1252<char>(), 0x201A, "\x82");
        test_fill(strf::iso_8859_1<char>(), 0x82, "\x82");
        test_fill(strf::iso_8859_3<char>(), 0x02D8, "\xA2");
        test_fill(strf::iso_8859_15<char>(), 0x20AC, "\xA4");

        std::vector<strf::encoding<char>> vec = { strf::windows_1252<char>()
                                                , strf::iso_8859_1<char>()
                                                , strf::iso_8859_3<char>()
                                                , strf::iso_8859_15<char>() };
        for (auto enc : vec)
        {
            test_fill(enc, 'a' , "a");
            test_fill(enc, 0x800, "?");
            test_invalid_fill_stop(enc, 0x800);
            test_invalid_fill_ignore(enc, 0x800);
        }
    }

    return boost::report_errors();
}
