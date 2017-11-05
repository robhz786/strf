//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"

#define TEST(EXPECTED) make_tester((EXPECTED), __FILE__, __LINE__)

#define TEST_RF(EXPECTED, RF) make_tester((EXPECTED), __FILE__, __LINE__, RF)

#define TEST_ERR(EXPECTED, ERR) make_tester((EXPECTED), __FILE__, __LINE__, ERR)

#define TEST_ERR_RF(EXPECTED, ERR, RF) make_tester((EXPECTED), __FILE__, __LINE__, ERR, RF)

namespace strf = boost::stringify::v0;

int main()
{
    {
        TEST(u8"--\u0080--\u07ff--\u0800--\uffff--\U00010000--\U0010ffff") =
            {
                "--", U'\u0080',
                "--", U'\u07ff',
                "--", U'\u0800',
                "--", U'\uffff',
                "--", U'\U00010000',
                "--", U'\U0010ffff',
            };

    }

    {   // defaul error handling: replace codepoints, by '\uFFFD'
        TEST_RF(u8"--\uFFFD--\uFFFD--\uFFFD--\uFFFD--\uFFFD\uFFFD\uFFFD--", 1.5) =
            {
                "--", (char32_t)0xD800,
                "--", (char32_t)0xDBFF,
                "--", (char32_t)0xDC00,
                "--", (char32_t)0xDFFF,
                "--", {(char32_t)0x110000, {"", 3}},
                "--", {(char32_t)0x110000, {"", 0}}
            };
    }


    {   // replace invalid codepoints by '?'

        auto facet = strf::make_u8encoder
            ( [](auto& ow, auto count) -> bool { return ow.repeat(count, 'X'); } );

        TEST_RF(u8"------X------X------X------X------XXX------", 1.5) .with(facet) =
            {
                "------", (char32_t)0xD800,
                "------", (char32_t)0xDBFF,
                "------", (char32_t)0xDC00,
                "------", (char32_t)0xDFFF,
                "------", {(char32_t)0x110000, {"", 3}},
                "------", {(char32_t)0x110000, {"", 0}},
            };
    }


    {   // emit error code on invalid codepoints
        TEST_ERR("------", std::make_error_code(std::errc::illegal_byte_sequence))
            .with(strf::make_u8encoder(strf::from_utf32_set_error_code<char>))
            = { "------", (char32_t)0x110000 };
    }


    {  // throw exception on invalid codepoints
        std::exception_ptr eptr;
        try
        {
            auto facet = strf::make_u8encoder(strf::from_utf32_throw<char>);
            auto rstr = strf::make_string .with(facet) = { (char32_t)0x110000 };
        }
        catch(...)
        {
            eptr = std::current_exception();
        }
        BOOST_TEST(eptr);
    }

    {   // WTF8 ( tolerate surrogates )
        auto facet = strf::make_u8encoder().wtf8();

        TEST("--\xED\xA0\x80--\xED\xAF\xBF--\xED\xB0\x80--\xED\xBF\xBF") .with(facet) =
            {
                "--", (char32_t)0xD800,
                "--", (char32_t)0xDBFF,
                "--", (char32_t)0xDC00,
                "--", (char32_t)0xDFFF
            };
    }

    // {   // MUTF8
    //     auto facet = strf::make_u8encoder().mutf8();

    //     TEST("--\xc0\x80--") .with(facet) = { "--", U'\0', "--" };
    // }

    {   // NO MUTF8
        auto rstr = strf::make_string [{"--", U'\0', "--"}];
        BOOST_TEST(rstr && rstr.value() == std::string("--\0--", 5));
    }

    return report_errors() || boost::report_errors();
}
