//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <boost/detail/lightweight_test.hpp>
#include "test_utils.hpp"
#include "error_code_emitter_arg.hpp"
#include "exception_thrower_arg.hpp"
#include <boost/stringify.hpp>

namespace strf = boost::stringify::v0;

template <typename CharT>
void basic_test()
{
    CharT result[100];
    std::fill(result, result + 100, CharT('-'));
    std::basic_string<CharT> expected;

    auto x = use_all_writing_function_of_output_writer
        ( strf::format(result)
        , expected );

    BOOST_TEST(x);
    BOOST_TEST(expected.length() == x.value());
    BOOST_TEST(expected == result);
}


template <typename CharT>
void test_array_too_small()
{
    CharT buff[3] = { 'a', 'a', 0 };
    auto x = strf::format(buff) ( 1234567 );

    BOOST_TEST(buff[0] == 0);
    BOOST_TEST(!x);
    BOOST_TEST(x.error() == std::errc::result_out_of_range);
}

template <typename CharT>
void test_informed_size_too_small()
{
    CharT buff[100] = { 'a', 'a', 0 };
    auto x = strf::format(buff, 3) ( 1234567 );

    BOOST_TEST(buff[0] == 0);
    BOOST_TEST(! x);
    BOOST_TEST(x.error() == std::errc::result_out_of_range);
}

template <typename CharT>
void test_informed_end_too_close()
{
    CharT buff[100] = { 'a', 'a', 0 };
    auto x = strf::format(buff, &buff[3]) ( 1234567 );

    BOOST_TEST(buff[0] == 0);
    BOOST_TEST(! x);
    BOOST_TEST(x.error() == std::errc::result_out_of_range);
}

int main()
{
    basic_test<char>();
    basic_test<char16_t>();
    basic_test<char32_t>();
    basic_test<wchar_t>();

    {   // Test char_ptr_writer::set_error
        //
        // When set_error(some_err) is called, some_err is returned at the end
        
        char16_t result[200] = u"-----------------------------";

        auto x = strf::format(result)
            (u"abcd", error_code_emitter_arg, u"lkjlj");

        BOOST_TEST(result[0] == u'\0');
        BOOST_TEST(! x);
        BOOST_TEST(x.error() == std::errc::invalid_argument);
    }

    {  // When exception is thrown 

        char16_t result[200] = u"-----------------------------";
        try
        {
            (void) strf::format(result) (u"abcd", exception_thrower_arg, u"lkjlj");
        }
        catch(...)
        {
        }

        BOOST_TEST(result[0] == u'\0');
    }
    
    test_array_too_small<char>();
    test_array_too_small<char16_t>();
    test_array_too_small<char32_t>();
    test_array_too_small<wchar_t>();

    test_informed_size_too_small<char>();
    test_informed_size_too_small<char16_t>();
    test_informed_size_too_small<char32_t>();
    test_informed_size_too_small<wchar_t>();

    test_informed_end_too_close<char>();
    test_informed_end_too_close<char16_t>();
    test_informed_end_too_close<char32_t>();
    test_informed_end_too_close<wchar_t>();
   
    {   // When overflow happens in char_ptr_writer::put(str, count)

        char16_t result[200] = u"--------------------------------------------------";

        auto x = strf::format(result, 3) ( u"abc" );

        BOOST_TEST(result[0] == u'\0');
        BOOST_TEST(! x);    
        BOOST_TEST(x.error() == std::errc::result_out_of_range);
    }

    {   // When overflow happens in char_ptr_writer::put(ch)

        char16_t result[200] = u"--------------------------------------------------";

        auto x = strf::format(result, 3) ( u'a', u'b', u'c' );

        BOOST_TEST(result[0] == u'\0');
        BOOST_TEST(! x);
        BOOST_TEST(x.error() == std::errc::result_out_of_range);
    }

    
   {   // When overflow happens in char_ptr_writer::put(ch, count)

       char result[200] = "--------------------------------------------------";
       auto x = strf::format(result, 2) (strf::multi('x', 10));
       BOOST_TEST(result[0] == '\0');
       BOOST_TEST(! x);
       BOOST_TEST(x.error() == std::errc::result_out_of_range);
   }
   // {   // When overflow happens in char_ptr_writer::put(ch, ch, count)

   //     char result[3] = "";
   //     auto x = strf::format(result, 3)
   //         (strf::multi(U'\u0080', 2));

   //     BOOST_TEST(result[0] == '\0');
   //     BOOST_TEST(! x);
   //     BOOST_TEST(x.error() == std::errc::result_out_of_range);
   // }
   // {   // When overflow happens in char_ptr_writer::put(ch, ch, ch, count)

   //     char result[200] = "--------------------------------------------------";
   //     auto x = strf::format(result, 5)
   //         (strf::multi(U'\u0800', 2));

   //     BOOST_TEST(result[0] == '\0');
   //     BOOST_TEST(! x);
   //     BOOST_TEST(x.error() == std::errc::result_out_of_range);
   // }
   // {   // When overflow happens in char_ptr_writer::put(ch, ch, ch, ch, count)

   //     char result[200] = "--------------------------------------------------";
   //     auto x = strf::format(result, 7)
   //         (strf::multi(U'\U00010000', 2));

   //     BOOST_TEST(result[0] == '\0');
   //     BOOST_TEST(! x);
   //     BOOST_TEST(x.error() == std::errc::result_out_of_range);
   // }

    int rc = report_errors() || boost::report_errors();
    return rc;
}
