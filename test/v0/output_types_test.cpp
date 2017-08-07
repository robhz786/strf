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

    // write_to(char*)

    {
        char buff[200] = "";
        strf::write_to(buff).with(fmt) [{"abcd", 10, 11}];
        BOOST_TEST(std::string(buff) == "abcd0xa0xb");
    }
    {
        wchar_t buff[200] = L"";
        strf::write_to(buff).with(fmt) [{L"abcd", 10, 11}];
        BOOST_TEST(std::wstring(buff) == L"abcd0xa0xb");
    }
    {
        char16_t buff[200] = u"";
        strf::write_to(buff).with(fmt) [{u"abcd", 10, 11}];
        BOOST_TEST(std::u16string(buff) == u"abcd0xa0xb");
    }
    {
        char32_t buff[200] = U"";
        strf::write_to(buff).with(fmt) [{U"abcd", 10, 11}];
        BOOST_TEST(std::u32string(buff) == U"abcd0xa0xb");
    }
    {
        char16_t buff[200] = u"";
        strf::write_to<to_upper_char_traits<char16_t>> (buff)
            .with(fmt) [{u"abcd", 10, 11}];
        BOOST_TEST(std::u16string(buff) == u"ABCD0XA0XB");
    }

    // write_to(char*, char* end)

    {
        char buff[200] = "---------------------------------------";
        char* end = buff + 200;
        auto result = strf::write_to(buff, end) ({"abcd", 8}, 'e');
        BOOST_TEST(result.length == 9);
        BOOST_TEST(result.success);
        BOOST_TEST(std::string(buff) == "    abcde");
    }
    {
        char buff[200] = "aaa";
        auto result = strf::write_to(buff, buff) ({"abcd", 8}, 'e');
        BOOST_TEST(result.length == 0);
        BOOST_TEST(! result.success);
        BOOST_TEST(std::string(buff) == "aaa");
    }
    {
        char buff[200] = "---------------------------------------";
        auto result = strf::write_to(buff, buff+1) ({"abcd", 8}, 'e');
        BOOST_TEST(result.length == 0);
        BOOST_TEST(! result.success);
        BOOST_TEST(std::string(buff) == "");
    }
    
    {
        char buff[200] = "---------------------------------------";
        char* end = buff + 10;
        auto result = strf::write_to(buff, end) ({"abcd", 8}, 'e');
        BOOST_TEST(result.length == 9);
        BOOST_TEST(result.success);

        BOOST_TEST(std::string(buff) == "    abcde");
    }
    {
        char buff[200] = "---------------------------------------";
        char* end = buff + 9;

        auto result = strf::write_to(buff, end) ({"abcd", 8}, 'e');
        BOOST_TEST(result.length == 8);
        BOOST_TEST( ! result.success);
        BOOST_TEST(std::string(buff) == "    abcd");
    }
    {
        char buff[200] = "---------------------------------------";
        char* end = buff + 9;
        auto result = strf::write_to(buff, end) ({"abcde", 9});
        BOOST_TEST(result.length == 8);
        BOOST_TEST( ! result.success);
        BOOST_TEST(std::string(buff) == "    abcd");
    }
    {
        char buff[200] = "---------------------------------------";
        char* end = buff + 9;
        auto result = strf::write_to(buff, end) ({"abcde", {9, "<"}});
        BOOST_TEST(result.length == 8);
        BOOST_TEST( ! result.success);
        BOOST_TEST(std::string(buff) == "abcde   ");
    }
    
    {
        char16_t buff[200] = u"---------------------------------------";
        char16_t* end = buff + 200;
        strf::write_to<to_upper_char_traits<char16_t>>(buff, end).with(fmt)
            ({u"abcd", 8}, u'e');
        BOOST_TEST(std::u16string(buff) == u"    ABCDE");
    }
    

    // write_to(char*, std::size_t end)
    {
        char buff[] = "---------------------------------------";

        auto result = strf::write_to(buff, 9) ({"abcd", 8}, 'e');
        BOOST_TEST(result.length == 8);
        BOOST_TEST( ! result.success);
        BOOST_TEST(std::string(buff) == "    abcd");
    }
    {
        char16_t buff[] = u"---------------------------------------";

        auto result = strf::write_to(buff, 9) ({u"abcd", 8}, u'e');
        BOOST_TEST(result.length == 8);
        BOOST_TEST( ! result.success);
        BOOST_TEST(std::u16string(buff) == u"    abcd");
    }
    {
        char32_t buff[] = U"---------------------------------------";

        auto result = strf::write_to(buff, 9) ({U"abcd", 8}, U'e');
        BOOST_TEST(result.length == 8);
        BOOST_TEST( ! result.success);
        BOOST_TEST(std::u32string(buff) == U"    abcd");
    }
    {
        wchar_t buff[] = L"---------------------------------------";

        auto result = strf::write_to(buff, 9) ({L"abcd", 8}, L'e');
        BOOST_TEST(result.length == 8);
        BOOST_TEST( ! result.success);
        BOOST_TEST(std::wstring(buff) == L"    abcd");
    }


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

    {
        std::ostringstream oss;
        auto status = strf::write_to(oss.rdbuf()) .with(fmt) ({"abcd", 8}, 'e');
        BOOST_TEST(status.count == 9);
        BOOST_TEST(oss.str() == "    abcde");
    }
    {
        std::basic_ostringstream<char16_t> oss;
        auto status = strf::write_to(oss.rdbuf()) .with(fmt) ({u"abcd", 8}, u'e');
        BOOST_TEST(status.count == 9);
        BOOST_TEST(oss.str() == u"    abcde");
    }
    {
        std::basic_ostringstream<char32_t> oss;
        auto status = strf::write_to(oss.rdbuf()) .with(fmt) ({U"abcd", 8}, U'e');
        BOOST_TEST(status.count == 9);
        BOOST_TEST(oss.str() == U"    abcde");
    }
    {
        std::basic_ostringstream<wchar_t> oss;
        auto status = strf::write_to(oss.rdbuf()) .with(fmt) ({L"abcd", 8}, L'e');
        BOOST_TEST(status.count == 9);
        BOOST_TEST(oss.str() == L"    abcde");
    }
    
    {
        const char* filename = "stringify_tmptestfile.tmp";
        std::remove(filename);
        {
            std::FILE* file = fopen(filename, "w");
            auto result = strf::write_to(file) .with(fmt) ({"abcd", 8}, 'e');
            fclose(file);
            BOOST_TEST(result.success);
            BOOST_TEST(result.count == 9);
        }
        {
            std::ifstream file(filename);
            std::string line;
            std::getline(file, line);
            BOOST_TEST(line == "    abcde");
        }
        std::remove(filename);
    }
    {
        const char* filename = "stringify_tmptestfile.tmp";
        std::remove(filename);
        {
            std::FILE* file = fopen(filename, "w");
            auto result = strf::wwrite_to(file) .with(fmt) ({L"abcd", 8}, L'e');
            fclose(file);
            BOOST_TEST(result.success);
            BOOST_TEST(result.count == 9);
        }
        {
            std::wifstream file(filename);
            std::wstring line;
            std::getline(file, line);
            BOOST_TEST(line == L"    abcde");
        }
        std::remove(filename);
    }


    int rc = boost::report_errors();
    return rc;
}


