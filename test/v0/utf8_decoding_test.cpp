//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"

#define TEST(EXPECTED) make_tester((EXPECTED), __FILE__, __LINE__)
#define TEST_ERR(EXPECTED, ERR) make_tester((EXPECTED), __FILE__, __LINE__, ERR)

namespace strf = boost::stringify::v0;

bool put_X(strf::u32output& rec)
{
    return rec.put(U'X');
}

bool emit_illegal_byte_sequence(strf::u32output& rec)
{
    rec.set_error(std::make_error_code(std::errc::illegal_byte_sequence));
    return false;
}



auto replace_err_by_X = strf::make_u8decoder(put_X);

int main()
{
    // simple correct utf8 sample
    //
    TEST( U"\u0079 \u0080 \u07FF \u0800 \uFFFF \U00010000 \U0010FFFF") =
        {u8"\u0079 \u0080 \u07FF \u0800 \uFFFF \U00010000 \U0010FFFF"};

    
    // leading byte not followed by enough continuation bytes
    //
    TEST(U" \uFFFD  \uFFFD  \uFFFD ")   = {" \xC0  \xEF\xBF  \xF1\xBF\xBF "};
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD ") = {" \xC0""\xEF\xBF""\xF1\xBF\xBF\xC0 "};

    
    // string ends before the sequence is complete
    //
    TEST(U"\uFFFD\uFFFD\uFFFD") = {"\xC0", "\xEF\xBF", "\xF1\xBF\xBF"};
    

    // invalid leading byte
    //
    TEST(U"\uFFFD \uFFFD") = {"\xF5\xBF\xBF\xBF \xF5\xBF\xBF\xBF"};
    TEST(U"\uFFFD \uFFFD") = {"\xF6\xBF\xBF\xBF \xF6\xBF\xBF\xBF"};
    TEST(U"\uFFFD \uFFFD") = {"\xF7\xBF\xBF\xBF \xF7\xBF\xBF\xBF"};
    TEST(U"\uFFFD \uFFFD") = {"\xF8\xB1\xBF\xBF\xBF \xF8\xB1\xBF\xBF\xBF"};
    TEST(U"\uFFFD \uFFFD") = {"\xF9\xB1\xBF\xBF\xBF \xF9\xB1\xBF\xBF\xBF"};

    TEST(U"XXXXXXXXXXX").with(replace_err_by_X)
        = {"\xF5\xF6\xF7\xF8\xF9\xFA\xFB\xFC\xFD\xFE\xFF"};


    // codepoints greater than 10FFFF
    //
    //     10   00100000 00000000
    //   10001  00000000 00000000
    
    TEST(U"\U0010FFFF ") = {u8"\U0010FFFF "};
    TEST(U"\uFFFD ") = {"\xF4\x90\x80\x80 "};
    TEST(U"\uFFFD ") = {"\xF5\xBF\xBF\xBF "};
    TEST(U"\uFFFD ") = {"\xF6\xBF\xBF\xBF "};
    TEST(U"\uFFFD ") = {"\xF7\xBF\xBF\xBF "};
    TEST(U"\uFFFD ") = {"\xF8\xBF\xBF\xBF\xBF "};
    TEST(U"\uFFFD ") = {"\xF9\xBF\xBF\xBF\xBF "};
    TEST(U"\uFFFD ") = {"\xFA\xBF\xBF\xBF\xBF "};
    TEST(U"\uFFFD ") = {"\xFB\xBF\xBF\xBF\xBF "};
    TEST(U"\uFFFD ") = {"\xFC\xBF\xBF\xBF\xBF\xBF "};
    TEST(U"\uFFFD ") = {"\xFD\xBF\xBF\xBF\xBF\xBF "};
    TEST(U"\uFFFD ") = {"\xFE\xBF\xBF\xBF\xBF\xBF "};
    TEST(U"\uFFFD ") = {"\xFF\xBF\xBF\xBF\xBF\xBF "};
    //
    // again without the trailing space
    TEST(U"\U0010FFFF") = {u8"\U0010FFFF"};
    TEST(U"\uFFFD") = {"\xF4\x90\x80\x80"};
    TEST(U"\uFFFD") = {"\xF5\xBF\xBF\xBF"};
    TEST(U"\uFFFD") = {"\xF6\xBF\xBF\xBF"};
    TEST(U"\uFFFD") = {"\xF7\xBF\xBF\xBF"};
    TEST(U"\uFFFD") = {"\xF8\xBF\xBF\xBF\xBF"};
    TEST(U"\uFFFD") = {"\xF9\xBF\xBF\xBF\xBF"};
    TEST(U"\uFFFD") = {"\xFA\xBF\xBF\xBF\xBF"};
    TEST(U"\uFFFD") = {"\xFB\xBF\xBF\xBF\xBF"};
    TEST(U"\uFFFD") = {"\xFC\xBF\xBF\xBF\xBF\xBF"};
    TEST(U"\uFFFD") = {"\xFD\xBF\xBF\xBF\xBF\xBF"};
    TEST(U"\uFFFD") = {"\xFE\xBF\xBF\xBF\xBF\xBF"};
    TEST(U"\uFFFD") = {"\xFF\xBF\xBF\xBF\xBF\xBF"};
    

    const char* highsurr_D800 = "\xED\xA0\x80";
    const char* lowsurr_DFFF  = "\xED\xBF\xBF";
    const char* toobig_110000 = "\xF4\x90\x80\x80";
    const char* mtf8_null     = "\xC0\x80";
    const char* overlong_007F = "\xC1\xBF";
    const char* overlong_07FF = "\xE0\x9F\xBF";
    const char* overlong_FFFF = "\xF0\x8F\xBF\xBF";

    // many invalid code sequeces:
    TEST(U"XXXXXXX").with(replace_err_by_X) =
        {
            highsurr_D800,
            lowsurr_DFFF,
            toobig_110000,
            mtf8_null,
            overlong_007F,
            overlong_07FF,
            overlong_FFFF
        };


    {   // tolerate surrogates

        auto result = * strf::make_u32string
            .with(strf::make_u8decoder().wtf8())
            [{
                highsurr_D800,
                lowsurr_DFFF,
                toobig_110000,
                mtf8_null,
                overlong_007F,
                overlong_07FF,
                overlong_FFFF
            }];
        
        BOOST_TEST(result[0] == (char32_t)0xD800);
        BOOST_TEST(result[1] == (char32_t)0XDFFF);
        BOOST_TEST(result[2] == U'\uFFFD');
        BOOST_TEST(result[3] == U'\uFFFD');
        BOOST_TEST(result[4] == U'\uFFFD');
        BOOST_TEST(result[5] == U'\uFFFD');
        BOOST_TEST(result[6] == U'\uFFFD');
    }

    { // tolerate overlong sequences

        auto result = * strf::make_u32string
            .with(strf::make_u8decoder().tolerate_overlong())
            [{
                highsurr_D800,
                lowsurr_DFFF,
                toobig_110000,
                mtf8_null,
                overlong_007F,
                overlong_07FF,
                overlong_FFFF
            }];
        
        BOOST_TEST(result[0] == U'\uFFFD');
        BOOST_TEST(result[1] == U'\uFFFD');
        BOOST_TEST(result[2] == U'\uFFFD');
        BOOST_TEST(result[3] == U'\0');
        BOOST_TEST(result[4] == U'\u007F');
        BOOST_TEST(result[5] == U'\u07FF');
        BOOST_TEST(result[6] == U'\uFFFF');
    }

    { // Modified UTF8

        auto result = * strf::make_u32string
            .with(strf::make_u8decoder().mutf8())
            [{
                highsurr_D800,
                lowsurr_DFFF,
                toobig_110000,
                mtf8_null,
                overlong_007F,
                overlong_07FF,
                overlong_FFFF
            }];

        BOOST_TEST(result[0] == U'\uFFFD');
        BOOST_TEST(result[1] == U'\uFFFD');
        BOOST_TEST(result[2] == U'\uFFFD');
        BOOST_TEST(result[3] == U'\0');
        BOOST_TEST(result[4] == U'\uFFFD');
        BOOST_TEST(result[5] == U'\uFFFD');
        BOOST_TEST(result[6] == U'\uFFFD');
    }
    
    

    // an invalid byte sequece followed by continuation
    // bytes produce only one error:
    //
    TEST(U"\uFFFD ") =  { "\xFF\xBF\xBF\xBF\xBF " };     // after invalid leading byte
    TEST(U"\uFFFD ") =  { "\xED\xA0\x80\xBF\xBF " };     // after D800
    TEST(U"\uFFFD ") =  { "\xED\xBF\xBF\xBF\xBF " };     // after DFFF
    TEST(U"\uFFFD ") =  { "\xF4\x90\x80\x80\xBF\xBF " }; // after 110000
    TEST(U"\uFFFD ") =  { "\xC0\x80\xBF\xBF " };         // after overlong null
    TEST(U"\uFFFD ") =  { "\xC1\xBF\xBF\xBF " };         // after overlong 7F
    TEST(U"\uFFFD ") =  { "\xE0\x9F\xBF\xBF\xBF " };     // after overlong FF
    TEST(U"\uFFFD ") =  { "\xF0\x8F\xBF\xBF\xBF\xBF " }; // after overlong FFFF
    // now without the trailing space:
    TEST(U"\uFFFD") =  { "\xFF\xBF\xBF\xBF\xBF" };     // after invalid leading byte
    TEST(U"\uFFFD") =  { "\xED\xA0\x80\xBF\xBF" };     // after D800
    TEST(U"\uFFFD") =  { "\xED\xBF\xBF\xBF\xBF" };     // after DFFF
    TEST(U"\uFFFD") =  { "\xF4\x90\x80\x80\xBF\xBF" }; // after 110000
    TEST(U"\uFFFD") =  { "\xC0\x80\xBF\xBF" };         // after overlong null
    TEST(U"\uFFFD") =  { "\xC1\xBF\xBF\xBF" };         // after overlong 7F
    TEST(U"\uFFFD") =  { "\xE0\x9F\xBF\xBF\xBF" };     // after overlong FF
    TEST(U"\uFFFD") =  { "\xF0\x8F\xBF\xBF\xBF\xBF" }; // after overlong FFFF

   
    {   // emit error code on invalid sequece
        auto expected_error = std::make_error_code(std::errc::illegal_byte_sequence);
        
        TEST_ERR(U"blah blah ", expected_error)
            .with(strf::make_u8decoder(emit_illegal_byte_sequence))
            [{ "blah", " blah \xEF\xBF", "blah"}];
    }
    
    
    return report_errors();
}
