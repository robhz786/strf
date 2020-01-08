//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <vector>

template <typename CharT>
struct fixture
{
    static constexpr std::size_t buff_size = 100;
    CharT buff[buff_size];
    CharT* const dest_begin = buff;
    CharT* dest_it = buff;
    CharT* const dest_end = buff + buff_size;
};

template <typename CharT>
void test_char( strf::encoding<CharT> enc
                    , char32_t ch
                    , std::basic_string<CharT> encoded_char )
{
    BOOST_TEST_LABEL ( "encoding: ", enc.name()
                     , "; char: \\u'", strf::hex((unsigned)ch), '\'');
    CharT buff[100];

    auto it = enc.encode_char(buff, ch);

    BOOST_TEST_EQ(std::size_t(it - buff), encoded_char.size());
    BOOST_TEST(std::equal(encoded_char.begin(), encoded_char.end(), buff));
}

int main()
{
    {   // UTF-8

        test_char<char>(strf::utf8<char>(), 0x7F, "\x7F");
        test_char<char>(strf::utf8<char>(), 0x80, "\xC2\x80");
        test_char<char>(strf::utf8<char>(), 0x800, "\xE0\xA0\x80");
        test_char<char>(strf::utf8<char>(), 0xFFFF, "\xEF\xBF\xBF");
        test_char<char>(strf::utf8<char>(), 0x10000, "\xF0\x90\x80\x80");
        test_char<char>(strf::utf8<char>(), 0x10FFFF, "\xF4\x8F\xBF\xBF");
        test_char<char>(strf::utf8<char>(), 0x110000, "\xEF\xBF\xBD");
    }
    {   // UTF-16

        test_char<char16_t>(strf::utf16<char16_t>(), U'a', u"a");
        test_char<char16_t>(strf::utf16<char16_t>(), 0xFFFF, u"\uFFFF");
        test_char<char16_t>(strf::utf16<char16_t>(), 0x10000, u"\U00010000");
        test_char<char16_t>(strf::utf16<char16_t>(), 0x10FFFF, u"\U0010FFFF");
        test_char<char16_t>(strf::utf16<char16_t>(), 0x110000, u"\uFFFD");
    }
    {   // UTF-32

        test_char<char32_t>(strf::utf32<char32_t>(), U'a', U"a");
        test_char<char32_t>(strf::utf32<char32_t>(), 0xFFFF, U"\uFFFF");
        test_char<char32_t>(strf::utf32<char32_t>(), 0x10000, U"\U00010000");
        test_char<char32_t>(strf::utf32<char32_t>(), 0x10FFFF, U"\U0010FFFF");
    }
    {
        // single byte encodings
        test_char<char>(strf::windows_1252<char>(), 0x201A, "\x82");
        test_char<char>(strf::iso_8859_1<char>(), 0x82, "\x82");
        test_char<char>(strf::iso_8859_3<char>(), 0x02D8, "\xA2");
        test_char<char>(strf::iso_8859_15<char>(), 0x20AC, "\xA4");

        for (auto enc : { strf::windows_1252<char>()
                        , strf::iso_8859_1<char>()
                        , strf::iso_8859_3<char>()
                        , strf::iso_8859_15<char>() } )
        {
            test_char<char>(enc, 'a' , "a");
            test_char<char>(enc, 0x800 , "?");
        }
    }


    return boost::report_errors();
}
