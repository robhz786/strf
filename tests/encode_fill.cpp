//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <strf/to_string.hpp>

template <typename CharT>
static STRF_TEST_FUNC std::size_t repeat
    ( CharT* dest
    , std::size_t dest_size
    , std::size_t count
    , strf::detail::simple_string_view<CharT> str )
{
    strf::basic_cstr_writer<CharT> ob{dest, dest_size};
    while (count--) {
        strf::to(ob) (str);
    }
    auto result = ob.finish();
    TEST_FALSE(result.truncated);
    return result.ptr - dest;
}

template <typename CharT, std::size_t DestSize>
inline STRF_TEST_FUNC std::size_t repeat
    ( CharT (&dest)[DestSize]
    , std::size_t count
    , strf::detail::simple_string_view<CharT> str )
{
    return repeat(dest, DestSize, count, str);
}

template <typename CharT, typename Encoding>
void STRF_TEST_FUNC test_fill
    ( Encoding enc
    , char32_t fill_char
    , strf::detail::simple_string_view<CharT> encoded_char )
{
    TEST_SCOPE_DESCRIPTION( enc.name(), ", test_fill_char: U+"
                          , strf::hex((unsigned)fill_char) );

    {
        constexpr std::int16_t count = 80;
        constexpr std::int16_t buffers_size = count * 6 + 2;
        CharT buff_result[buffers_size];
        auto res = strf::to(buff_result).with(enc) (strf::right(CharT('x'), count + 1, fill_char));
        auto result_len = res.ptr - buff_result;

        CharT buff_expected_fill[buffers_size];
        auto expected_fill_len = repeat(buff_expected_fill, count, encoded_char);
        TEST_EQ(expected_fill_len + 1, result_len);
        TEST_TRUE(buff_result[result_len - 1] == CharT('x'));
        TEST_TRUE(strf::detail::str_equal(buff_result, buff_expected_fill, expected_fill_len));
    }
}

void STRF_TEST_FUNC test_encode_fill()
{
    {
        // UTF-8

        test_fill<char>(strf::utf8<char>(), 0x7F, "\x7F");
        test_fill<char>(strf::utf8<char>(), 0x80, "\xC2\x80");
        test_fill<char>(strf::utf8<char>(), 0x800, "\xE0\xA0\x80");
        test_fill<char>(strf::utf8<char>(), 0xFFFF, "\xEF\xBF\xBF");
        test_fill<char>(strf::utf8<char>(), 0x10000, "\xF0\x90\x80\x80");
        test_fill<char>(strf::utf8<char>(), 0x10FFFF, "\xF4\x8F\xBF\xBF");
        test_fill<char>(strf::utf8<char>(), 0x110000, "\xEF\xBF\xBD");
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
        test_fill<char>(strf::windows_1252<char>(), 0x201A, "\x82");
        test_fill<char>(strf::iso_8859_1<char>(), 0x82, "\x82");
        test_fill<char>(strf::iso_8859_3<char>(), 0x02D8, "\xA2");
        test_fill<char>(strf::iso_8859_15<char>(), 0x20AC, "\xA4");

        test_fill<char>(strf::ascii<char>(), 'a' , "a");
        test_fill<char>(strf::windows_1252<char>(), 'a' , "a");
        test_fill<char>(strf::iso_8859_1<char>()  , 'a' , "a");
        test_fill<char>(strf::iso_8859_3<char>()  , 'a' , "a");
        test_fill<char>(strf::iso_8859_15<char>() , 'a' , "a");

        test_fill<char>(strf::ascii<char>(), 0x800, "?");
        test_fill<char>(strf::windows_1252<char>(), 0x800, "?");
        test_fill<char>(strf::iso_8859_1<char>()  , 0x800, "?");
        test_fill<char>(strf::iso_8859_3<char>()  , 0x800, "?");
        test_fill<char>(strf::iso_8859_15<char>() , 0x800, "?");
    }

}
