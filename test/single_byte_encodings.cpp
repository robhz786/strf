//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

#include <string>
#include <algorithm>
#include <strf.hpp>

std::string make_str_0_to_xff()
{
    std::string str(0x100, '\0');
    for(unsigned i = 0; i < 0x100; ++i)
        str[i] = static_cast<char>(i);
    return str;
}

const std::string str_0_to_xff = make_str_0_to_xff();

std::string char_0_to_0xff_sanitized(strf::encoding<char> enc)
{
    std::string str;
    for(unsigned i = 0; i < 0x100; ++i)
    {
        char32_t ch32 = enc.decode_single_char(static_cast<std::uint8_t>(i));
        unsigned char ch = ( ch32 == (char32_t)-1
                           ? static_cast<unsigned char>('?')
                           : static_cast<unsigned char>(i) );
        str.push_back(ch);
    }
    return str;
}

void test(const strf::encoding<char>& enc, std::u32string decoded_0_to_0x100)
{
    BOOST_TEST_LABEL (enc.name());

    {
        // to UTF-32
        TEST(decoded_0_to_0x100) (strf::sani(str_0_to_xff, enc));
    }

    std::u32string valid_u32input = decoded_0_to_0x100;
    valid_u32input.erase( std::remove( valid_u32input.begin()
                                     , valid_u32input.end()
                                     , U'\uFFFD' )
                        , valid_u32input.end() );
    {
        // from and back to UTF-32
        auto enc_str = strf::to_string.with(enc) (strf::sani(valid_u32input));
        auto u32str = strf::to_u32string (strf::sani(enc_str, enc));
        BOOST_TEST(u32str == valid_u32input);
    }
    {
        // from UTF-8
        auto u8str = strf::to_string (strf::sani(valid_u32input));
        auto enc_str = strf::to_string.with(enc) (strf::sani(u8str, strf::utf8<char>()));
        auto u32str = strf::to_u32string (strf::sani(enc_str, enc));
        BOOST_TEST(u32str == valid_u32input);

    }
    {   // from UTF-8
        auto u8str = strf::to_string(strf::sani(decoded_0_to_0x100));
        TEST(char_0_to_0xff_sanitized(enc))
            .with(enc)
            (strf::sani(u8str, strf::utf8<char>()));
    }

    TEST(char_0_to_0xff_sanitized(enc)).with(enc) (strf::sani(str_0_to_xff));
    TEST("---?+++")
        .with(enc, strf::encoding_error::replace)
        (strf::sani(u"---\U0010FFFF+++"));

#if defined(__cpp_exceptions)

    {
        auto facets = strf::pack(enc, strf::encoding_error::stop);
        BOOST_TEST_THROWS(
            ( (strf::to_string.with(facets)(strf::sani(u"---\U0010FFFF++"))))
            , strf::encoding_failure );
    }

#endif // defined(__cpp_exceptions)
}


std::u32string decoded_0_to_xff_iso_8859_1()
{
    std::u32string table(0x100, u'\0');
    unsigned i = 0;
    for (char32_t& ch : table)
    {
        ch = i++;
    }
    return table;
}

std::u32string decoded_0_to_xff_iso_8859_3()
{
    std::u32string table;
    for(unsigned i = 0; i < 0xA1; ++i)
    {
        table.push_back(i);
    }
    table.append(U"\u0126\u02D8\u00A3\u00A4\uFFFD\u0124\u00A7"
                 U"\u00A8\u0130\u015E\u011E\u0134\u00AD\uFFFD\u017B"
                 U"\u00B0\u0127\u00B2\u00B3\u00B4\u00B5\u0125\u00B7"
                 U"\u00B8\u0131\u015F\u011F\u0135\u00BD\uFFFD\u017C"
                 U"\u00C0\u00C1\u00C2\uFFFD\u00C4\u010A\u0108\u00C7"
                 U"\u00C8\u00C9\u00CA\u00CB\u00CC\u00CD\u00CE\u00CF"
                 U"\uFFFD\u00D1\u00D2\u00D3\u00D4\u0120\u00D6\u00D7"
                 U"\u011C\u00D9\u00DA\u00DB\u00DC\u016C\u015C\u00DF"
                 U"\u00E0\u00E1\u00E2\uFFFD\u00E4\u010B\u0109\u00E7"
                 U"\u00E8\u00E9\u00EA\u00EB\u00EC\u00ED\u00EE\u00EF"
                 U"\uFFFD\u00F1\u00F2\u00F3\u00F4\u0121\u00F6\u00F7"
                 U"\u011D\u00F9\u00FA\u00FB\u00FC\u016D\u015D\u02D9");

    return table;
}

std::u32string decoded_0_to_xff_iso_8859_15()
{
    auto table =  decoded_0_to_xff_iso_8859_1();
    table[0xA4] = 0x20AC;
    table[0xA6] = 0x0160;
    table[0xA8] = 0x0161;
    table[0xB4] = 0x017D;
    table[0xB8] = 0x017E;
    table[0xBC] = 0x0152;
    table[0xBD] = 0x0153;
    table[0xBE] = 0x0178;
    return table;
}

std::u32string decoded_0_to_xff_windows_1252()
{
    auto table = decoded_0_to_xff_iso_8859_1();
    const char32_t r80_to_9F[] =
        { 0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021
        , 0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0x008D, 0x017D, 0x008F
        , 0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014
        , 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x009D, 0x017E, 0x0178 };
    std::copy(r80_to_9F, r80_to_9F + 0x20, table.begin() + 0x80);
    return table;
}

int main()
{
    test(strf::iso_8859_1<char>(), decoded_0_to_xff_iso_8859_1());
    test(strf::iso_8859_3<char>(), decoded_0_to_xff_iso_8859_3());
    test(strf::iso_8859_15<char>(), decoded_0_to_xff_iso_8859_15());
    test(strf::windows_1252<char>(), decoded_0_to_xff_windows_1252() );
    return boost::report_errors();
}
