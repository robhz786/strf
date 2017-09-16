//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <boost/detail/lightweight_test.hpp>
#include "test_utils.hpp"
#include <boost/stringify.hpp>
#include <limits>
#include <locale>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string.h>



template <typename T> struct is_long :  public std::is_same<long, T>
{
};

int main()
{
    namespace strf = boost::stringify::v0;

    auto fmt = strf::make_ftuple(strf::showbase, strf::hex);

    // append_to(std::string)

    {
        std::string str = "qwert--";
        strf::append_to(str) .with(fmt) ("abcd", 10, 11);
        BOOST_TEST(str == "qwert--abcd0xa0xb");
    }
    {
        std::wstring str = L"qwert--";
        strf::append_to(str) .with(fmt) (L"abcd", 10, 11);
        BOOST_TEST(str == L"qwert--abcd0xa0xb");
    }
    {
        std::u16string str = u"qwert--";
        strf::append_to(str) .with(fmt) (u"abcd", 10, 11);
        BOOST_TEST(str == u"qwert--abcd0xa0xb");
    }
    {
        std::u32string str = U"qwert--";
        strf::append_to(str) .with(fmt) (U"abcd", 10, 11);
        BOOST_TEST(str == U"qwert--abcd0xa0xb");
    }
    {
        std::basic_string<char16_t, to_upper_char_traits<char16_t>> str{u"qwert--"};
        strf::append_to(str) .with(fmt) (u"abcd", 10, 11);
        BOOST_TEST(std::u16string(str.c_str()) == u"QWERT--ABCD0XA0XB");
    }

    // assign_to(std::string)

    {
        std::string str = "qwert--";
        strf::assign_to(str) .with(fmt) ("abcd", 10, 11);
        BOOST_TEST(str == "abcd0xa0xb");
    }
    {
        std::wstring str = L"qwert--";
        strf::assign_to(str) .with(fmt) (L"abcd", 10, 11);
        BOOST_TEST(str == L"abcd0xa0xb");
    }
    {
        std::u16string str = u"qwert--";
        strf::assign_to(str) .with(fmt) (u"abcd", 10, 11);
        BOOST_TEST(str == u"abcd0xa0xb");
    }
    {
        std::u32string str = U"qwert--";
        strf::assign_to(str) .with(fmt) (U"abcd", 10, 11);
        BOOST_TEST(str == U"abcd0xa0xb");
    }
    {
        std::basic_string<char16_t, to_upper_char_traits<char16_t>> str{u"qwert--"};
        strf::assign_to(str) .with(fmt) (u"abcd", 10, 11);
        BOOST_TEST(std::u16string(str.c_str()) == u"ABCD0XA0XB");
    }

    // make_string

    {
        auto str = strf::make_string
            .with(strf::showpos, strf::noshowpos_if<is_long>)
            ("abcd", (long)0, 1, 2, {3, "-"}, {(long)4, "+"});

        BOOST_TEST(str == "abcd0+1+23+4");
    }
    {
        auto str = strf::make_string
            .with(strf::showpos, strf::noshowpos_if<is_long>)
            ("abcd", (long)0, 1, 2, {3, "-"}, {(long)4, "+"});

        BOOST_TEST(str == "abcd0+1+23+4");
    }
    {
        auto str = strf::make_u16string
            .with(strf::showpos, strf::noshowpos_if<is_long>)
            (u"abcd", (long)0, 1, 2, {3, "-"}, {(long)4, "+"});

        BOOST_TEST(str == u"abcd0+1+23+4");
    }
    {
        auto str = strf::make_u32string
            .with(strf::showpos, strf::constrain<is_long>(strf::noshowpos))
            (U"abcd", (long)0, 1, 2, {3, "-"}, {(long)4, "+"});

        BOOST_TEST(str == U"abcd0+1+23+4");
    }
    {
        auto str = strf::make_wstring
            .with(strf::showpos, strf::noshowpos_if<is_long>)
            (L"abcd", (long)0, 1, 2, {3, "-"}, {(long)4, "+"});

        BOOST_TEST(str == L"abcd0+1+23+4");
    }
    {
        auto str = strf::make_basic_string<char16_t, to_upper_char_traits<char16_t>>
            .with(strf::showbase, strf::hex)
            (u"abcd", 11);

        BOOST_TEST(std::u16string(str.c_str()) == u"ABCD0XB");
    }

    int rc = boost::report_errors();
    return rc;
}


