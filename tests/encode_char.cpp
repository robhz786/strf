//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

template <typename CharT>
struct fixture
{
    static constexpr std::size_t buff_size = 100;
    CharT buff[buff_size];
    CharT* const dest_begin = buff;
    CharT* dest_it = buff;
    CharT* const dest_end = buff + buff_size;
};

template <typename CharT, typename Charset>
STRF_TEST_FUNC void test_char
    ( Charset charset
    , char32_t ch
    , strf::detail::simple_string_view<CharT> encoded_char )
{
    TEST_SCOPE_DESCRIPTION( "encoding: ", charset.name()
                          , "; char: \\u'", strf::hex(static_cast<unsigned>(ch)), '\'');

    CharT buff[100];
    auto *it = charset.encode_char(buff, ch);

    TEST_EQ(std::size_t(it - buff), encoded_char.size());
    // clang-tidy says the pointer inside test_utils::test_scope::first_test_scope_()
    // is dangling, while I think is not.
    // Anyway, even if it is dangling, that's not a bug in the library itself,
    // just in the test.
    // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
    TEST_TRUE(strf::detail::str_equal(encoded_char.data(), buff, encoded_char.size()));
} // NOLINT(clang-analyzer-core.StackAddressEscape)

STRF_TEST_FUNC void test_encode_char()
{
    {   // UTF-8

        test_char<char>(strf::utf_t<char>(), 0x7F, "\x7F");
        test_char<char>(strf::utf_t<char>(), 0x80, "\xC2\x80");
        test_char<char>(strf::utf_t<char>(), 0x800, "\xE0\xA0\x80");
        test_char<char>(strf::utf_t<char>(), 0xFFFF, "\xEF\xBF\xBF");
        test_char<char>(strf::utf_t<char>(), 0x10000, "\xF0\x90\x80\x80");
        test_char<char>(strf::utf_t<char>(), 0x10FFFF, "\xF4\x8F\xBF\xBF");
        test_char<char>(strf::utf_t<char>(), 0x110000, "\xEF\xBF\xBD");
    }
    {   // UTF-16

        test_char<char16_t>(strf::utf_t<char16_t>(), U'a', u"a");
        test_char<char16_t>(strf::utf_t<char16_t>(), 0xFFFF, u"\uFFFF");
        test_char<char16_t>(strf::utf_t<char16_t>(), 0x10000, u"\U00010000");
        test_char<char16_t>(strf::utf_t<char16_t>(), 0x10FFFF, u"\U0010FFFF");
        test_char<char16_t>(strf::utf_t<char16_t>(), 0x110000, u"\uFFFD");
    }
    {   // UTF-32

        test_char<char32_t>(strf::utf_t<char32_t>(), U'a', U"a");
        test_char<char32_t>(strf::utf_t<char32_t>(), 0xFFFF, U"\uFFFF");
        test_char<char32_t>(strf::utf_t<char32_t>(), 0x10000, U"\U00010000");
        test_char<char32_t>(strf::utf_t<char32_t>(), 0x10FFFF, U"\U0010FFFF");
    }
    {
        // single byte encodings
        test_char<char>(strf::windows_1252_t<char>(), 0x201A, "\x82");
        test_char<char>(strf::iso_8859_1_t<char>(), 0x82, "\x82");
        test_char<char>(strf::iso_8859_3_t<char>(), 0x02D8, "\xA2");
        test_char<char>(strf::iso_8859_15_t<char>(), 0x20AC, "\xA4");

        test_char<char>(strf::ascii_t<char>()       , 'a' , "a");
        test_char<char>(strf::windows_1252_t<char>(), 'a' , "a");
        test_char<char>(strf::iso_8859_1_t<char>()  , 'a' , "a");
        test_char<char>(strf::iso_8859_3_t<char>()  , 'a' , "a");
        test_char<char>(strf::iso_8859_15_t<char>() , 'a' , "a");

        test_char<char>(strf::ascii_t<char>()       , 0x800 , "?");
        test_char<char>(strf::windows_1252_t<char>(), 0x800 , "?");
        test_char<char>(strf::iso_8859_1_t<char>()  , 0x800 , "?");
        test_char<char>(strf::iso_8859_3_t<char>()  , 0x800 , "?");
        test_char<char>(strf::iso_8859_15_t<char>() , 0x800 , "?");
    }
}

REGISTER_STRF_TEST(test_encode_char)
