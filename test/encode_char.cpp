//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "lightweight_test_label.hpp"
#include <boost/stringify.hpp>
#include <vector>

namespace strf = boost::stringify::v0;

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
void test_valid_char( strf::encoding<CharT> enc
                    , char32_t ch
                    , std::basic_string<CharT> encoded_char )
{
    BOOST_TEST_LABEL << "encoding: " << enc.name() << "; char: \\u'"
                     << std::hex << (unsigned)ch << '\'' << std::dec;

    CharT buff[100];

    auto it = enc.encode_char(buff, ch);

    BOOST_TEST_EQ(std::size_t(it - buff), encoded_char.size());
    BOOST_TEST(std::equal(encoded_char.begin(), encoded_char.end(), buff));
}

template <typename CharT>
void test_invalid_char( strf::encoding<CharT> enc, char32_t ch )
{
    BOOST_TEST_LABEL << "encoding: " << enc.name() << "; char: \\u'"
                     << std::hex << (unsigned)ch << '\'' << std::dec;

    CharT buff[100];
    auto it = enc.encode_char( buff, ch );

    BOOST_TEST_EQ(it, buff);
}



int main()
{
    {   // UTF-8

        test_valid_char<char>(strf::utf8(), 0x7F, "\x7F");
        test_valid_char<char>(strf::utf8(), 0x80, "\xC2\x80");
        test_valid_char<char>(strf::utf8(), 0x800, "\xE0\xA0\x80");
        test_valid_char<char>(strf::utf8(), 0xFFFF, "\xEF\xBF\xBF");
        test_valid_char<char>(strf::utf8(), 0x10000, "\xF0\x90\x80\x80");
        test_valid_char<char>(strf::utf8(), 0x10FFFF, "\xF4\x8F\xBF\xBF");

        test_invalid_char(strf::utf8(), 0x110000);
    }
    {   // UTF-16

        test_valid_char<char16_t>(strf::utf16(), U'a', u"a");
        test_valid_char<char16_t>(strf::utf16(), 0xFFFF, u"\uFFFF");
        test_valid_char<char16_t>(strf::utf16(), 0x10000, u"\U00010000");
        test_valid_char<char16_t>(strf::utf16(), 0x10FFFF, u"\U0010FFFF");

           test_invalid_char(strf::utf16(), 0x110000);
    }
    {   // UTF-32

        test_valid_char<char32_t>(strf::utf32(), U'a', U"a");
        test_valid_char<char32_t>(strf::utf32(), 0xFFFF, U"\uFFFF");
        test_valid_char<char32_t>(strf::utf32(), 0x10000, U"\U00010000");
        test_valid_char<char32_t>(strf::utf32(), 0x10FFFF, U"\U0010FFFF");
    }
    {
        // single byte encodings
        test_valid_char<char>(strf::windows_1252(), 0x201A, "\x82");
        test_valid_char<char>(strf::iso_8859_1(), 0x82, "\x82");
        test_valid_char<char>(strf::iso_8859_3(), 0x02D8, "\xA2");
        test_valid_char<char>(strf::iso_8859_15(), 0x20AC, "\xA4");

        for (auto enc : { strf::windows_1252()
                        , strf::iso_8859_1()
                        , strf::iso_8859_3()
                        , strf::iso_8859_15() } )
        {
            test_valid_char<char>(enc, 'a' , "a");
            test_invalid_char(enc, 0x800);
            test_invalid_char(enc, 0x800);
        }
    }


    return boost::report_errors();
}
