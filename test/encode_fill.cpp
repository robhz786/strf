//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <vector>

template <typename CharT>
std::basic_string<CharT> repeat(std::size_t count, std::basic_string<CharT> str)
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

template <typename CharT, typename Encoding>
void test_fill
    ( Encoding enc, char32_t fill_char, std::basic_string<CharT> encoded_char )
{
    TEST_SCOPE_DESCRIPTION( enc.name(), ", test_fill_char: U+"
                          , strf::hex((unsigned)fill_char) );

    {
        std::int16_t count = 10;
        auto result = strf::to_basic_string<CharT>.with(enc)
            (strf::right(CharT('x'), 11, fill_char));

        auto expected = repeat(count, encoded_char);
        expected.push_back(CharT('x'));

        TEST_TRUE(result == expected);
    }
    {
        std::int16_t count = 200;
        auto result = strf::to_basic_string<CharT>.with(enc)
            (strf::right(CharT('x'), count + 1, fill_char));

        auto expected = repeat(count, encoded_char);
        expected.push_back(CharT('x'));

        TEST_TRUE(result == expected);
    }
}

template <typename CharT, typename Encoding>
inline void test_fill(Encoding enc, char32_t fill_char, const CharT* encoded_char)
{
    return test_fill(enc, fill_char, std::basic_string<CharT>{encoded_char});
}

int main()
{
    {
        // UTF-8

        test_fill(strf::utf8<char>(), 0x7F, "\x7F");
        test_fill(strf::utf8<char>(), 0x80, "\xC2\x80");
        test_fill(strf::utf8<char>(), 0x800, "\xE0\xA0\x80");
        test_fill(strf::utf8<char>(), 0xFFFF, "\xEF\xBF\xBF");
        test_fill(strf::utf8<char>(), 0x10000, "\xF0\x90\x80\x80");
        test_fill(strf::utf8<char>(), 0x10FFFF, "\xF4\x8F\xBF\xBF");

        test_fill(strf::utf8<char>(), 0x110000, "\xEF\xBF\xBD");
    }

    {
        // UTF-16;
        // test_fill(strf::utf16<char16_t>(), U'a', u"a");
        test_fill<char16_t>(strf::utf16<char16_t>(), 0x10000,  u"\U00010000");
        test_fill<char16_t>(strf::utf16<char16_t>(), 0x10FFFF, u"\U0010FFFF");
        test_fill<char16_t>(strf::utf16<char16_t>(), 0x110000, u"\uFFFD");
    }

    {
        // UTF-32;
        test_fill<char32_t>( strf::utf32<char32_t>(), U'a', U"a");
        test_fill<char32_t>(strf::utf32<char32_t>(), 0x10000,  U"\U00010000");
        test_fill<char32_t>(strf::utf32<char32_t>(), 0x10FFFF, U"\U0010FFFF");

        //test_fill<char32_t>(strf::utf32<char32_t>(), 0x110000, U"\uFFFD");
    }

    {
        // single byte encodings
        test_fill(strf::windows_1252<char>(), 0x201A, "\x82");
        test_fill(strf::iso_8859_1<char>(), 0x82, "\x82");
        test_fill(strf::iso_8859_3<char>(), 0x02D8, "\xA2");
        test_fill(strf::iso_8859_15<char>(), 0x20AC, "\xA4");

        test_fill(strf::ascii<char>(), 'a' , "a");
        test_fill(strf::windows_1252<char>(), 'a' , "a");
        test_fill(strf::iso_8859_1<char>()  , 'a' , "a");
        test_fill(strf::iso_8859_3<char>()  , 'a' , "a");
        test_fill(strf::iso_8859_15<char>() , 'a' , "a");

        test_fill(strf::ascii<char>(), 0x800, "?");
        test_fill(strf::windows_1252<char>(), 0x800, "?");
        test_fill(strf::iso_8859_1<char>()  , 0x800, "?");
        test_fill(strf::iso_8859_3<char>()  , 0x800, "?");
        test_fill(strf::iso_8859_15<char>() , 0x800, "?");
    }

    return test_finish();
}
