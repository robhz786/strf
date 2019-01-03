//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "lightweight_test_label.hpp"

#include <boost/utility/string_view.hpp>
#include <boost/stringify/v0/detail/transcoding.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/tuple.hpp>
#include <algorithm>

namespace strf = boost::stringify::v0;
namespace hana = boost::hana;

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN;

inline std::ostream& operator<<(std::ostream& dest, strf::cv_result r)
{
    return dest << ( r == strf::cv_result::success ? "success"
                   : r == strf::cv_result::insufficient_space ? "insufficient_space"
                   : r == strf::cv_result::invalid_char ? "invalid_char"
                   : "???" );
}

BOOST_STRINGIFY_V0_NAMESPACE_END;

template <typename CharT>
bool str_equal(const CharT* a, const CharT* b, std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        if(a[i] != b[i])
        {
            return false;
        }
    }
    return true;
    //return std::char_traits<CharT>::compare(a, b, count) == 0;
}


void fill_iso_8859_1_table(char32_t* table)
{
    for(unsigned i = 0; i < 0x100; ++i)
    {
        table[i] = i;
    }
}

void fill_iso_8859_3_table(char32_t* table)
{
    for(unsigned i = 0; i < 0xA1; ++i)
    {
        table[i] = i;
    }

    char32_t undef = 0xFFFD;

    char32_t table2[] =
        {/*    */ 0x0126, 0x02D8, 0x00A3, 0x00A4, undef,  0x0124, 0x00A7
        , 0x00A8, 0x0130, 0x015E, 0x011E, 0x0134, 0x00AD,  undef, 0x017B
        , 0x00B0, 0x0127, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x0125, 0x00B7
        , 0x00B8, 0x0131, 0x015F, 0x011F, 0x0135, 0x00BD,  undef, 0x017C
        , 0x00C0, 0x00C1, 0x00C2,  undef, 0x00C4, 0x010A, 0x0108, 0x00C7
        , 0x00C8, 0x00C9, 0x00CA, 0x00CB, 0x00CC, 0x00CD, 0x00CE, 0x00CF
        ,  undef, 0x00D1, 0x00D2, 0x00D3, 0x00D4, 0x0120, 0x00D6, 0x00D7
        , 0x011C, 0x00D9, 0x00DA, 0x00DB, 0x00DC, 0x016C, 0x015C, 0x00DF
        , 0x00E0, 0x00E1, 0x00E2,  undef, 0x00E4, 0x010B, 0x0109, 0x00E7
        , 0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x00EC, 0x00ED, 0x00EE, 0x00EF
        ,  undef, 0x00F1, 0x00F2, 0x00F3, 0x00F4, 0x0121, 0x00F6, 0x00F7
        , 0x011D, 0x00F9, 0x00FA, 0x00FB, 0x00FC, 0x016D, 0x015D, 0x02D9 };

    for(unsigned i = 0xA1; i < 0x100; ++i)
    {
        table[i] = table2[i - 0xA1];
    }
}

void fill_iso_8859_15_table(char32_t* table)
{
    fill_iso_8859_1_table(table);
    table[0xA4] = 0x20AC;
    table[0xA6] = 0x0160;
    table[0xA8] = 0x0161;
    table[0xB4] = 0x017D;
    table[0xB8] = 0x017E;
    table[0xBC] = 0x0152;
    table[0xBD] = 0x0153;
    table[0xBE] = 0x0178;
}

void fill_windows_1252_table(char32_t* table)
{
    fill_iso_8859_1_table(table);

    const char32_t r80_to_9F[] =
        { 0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021
        , 0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0x008D, 0x017D, 0x008F
        , 0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014
        , 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x009D, 0x017E, 0x0178 };

    std::copy(r80_to_9F, r80_to_9F + 0x20, table + 0x80);
}


typedef void (*fill_table_func)(char32_t*);

void test(const strf::encoding<char>& enc, fill_table_func fill_table)
{
    BOOST_TEST_LABEL << enc.name;

    char32_t u32table [0x101] = {0};
    u32table [0x100] = 0x100;
    char32_t* u32table_end = u32table + 0x100;
    fill_table(const_cast<char32_t*>(u32table));

    char  char_0_to_0xff[0x100] = {0};
    char* char_0_to_0xff_end = char_0_to_0xff + 0x100;
    for(unsigned  i = 0; i < 0x100; ++i)
    {
        char_0_to_0xff[i] = i;
    }

    char char_0_to_0xff_sanitized[0x100] = {0};
    for(unsigned i = 0; i < 0x100; ++i)
    {
        unsigned char ch = enc.decode_single_char(i) == (char32_t)-1 ? '?': i;
        char_0_to_0xff_sanitized[i] = ch;
    }


    {   // convert to UTF-32
        char32_t result [0x100];
        char32_t* result_end = result + 0x100;

        const auto* src = char_0_to_0xff;
        auto* dest = result;
        auto res = enc.to_u32.transcode( &src, char_0_to_0xff_end
                                       , &dest, result_end
                                       , strf::error_handling::replace, false );

        BOOST_TEST_EQ(res, strf::cv_result::success);
        BOOST_TEST_EQ(src, char_0_to_0xff_end);
        BOOST_TEST_EQ(dest, result_end);
        BOOST_TEST(str_equal(result, u32table, 0x100));

        auto size = enc.to_u32.necessary_size( char_0_to_0xff, char_0_to_0xff_end
                                             , strf::error_handling::replace
                                             , false );
        BOOST_TEST_EQ(size, 0x100);
    }

    {   // convert from UTF-32
        char result [0x101];
        char* result_end = result + 0x101;

        const auto* src = u32table;
        auto* dest = result;
        auto res = enc.from_u32.transcode( &src, u32table + 0x101
                                         , &dest, result_end
                                         , strf::error_handling::replace, false );

        BOOST_TEST_EQ(res, strf::cv_result::success);
        BOOST_TEST_EQ(src, u32table + 0x101);
        BOOST_TEST_EQ(dest, result + 0x101);
        BOOST_TEST(str_equal(result, char_0_to_0xff_sanitized, 0x100));
        BOOST_TEST_EQ(result[0x100], '?');

        auto size = enc.from_u32.necessary_size( u32table, u32table_end
                                               , strf::error_handling::replace
                                               , false );
        BOOST_TEST_EQ(size, 0x100);
    }

    {   // sanitize
        char result [0x100];
        char* result_end = result + 0x100;

        const auto* src = char_0_to_0xff;
        auto* dest = result;
        auto res = enc.sanitizer.transcode( &src, char_0_to_0xff_end
                                          , &dest, result_end
                                          , strf::error_handling::replace
                                          , false );
        BOOST_TEST_EQ(res, strf::cv_result::success);

        BOOST_TEST(str_equal(result, char_0_to_0xff_sanitized, 0x100));
        BOOST_TEST_EQ(src, char_0_to_0xff_end);
        BOOST_TEST_EQ(dest, result_end);

        auto size = enc.sanitizer.necessary_size( char_0_to_0xff, char_0_to_0xff_end
                                                , strf::error_handling::replace
                                                , false );
        BOOST_TEST_EQ(size, 0x100);
    }
}


int main()
{
    auto encodings = hana::make_tuple
        ( std::make_pair(strf::iso_8859_1(), fill_iso_8859_1_table)
        , std::make_pair(strf::iso_8859_3(), fill_iso_8859_3_table)
        , std::make_pair(strf::iso_8859_15(), fill_iso_8859_15_table)
        , std::make_pair(strf::windows_1252(), fill_windows_1252_table) );

    hana::for_each(encodings, [](const auto p){
            test(p.first, p.second);
        });

    return boost::report_errors();
}
