//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

STRF_TEST_FUNC void test_output_buffer_functions()
{

    {   // Cover strf::detail::output_buffer_interchar_copy

        const unsigned char sample[10] =
            { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'};

        {
            test_utils::recycle_call_tester<char> dst
                { {"abcdefghij", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 5} };

            strf::detail::output_buffer_interchar_copy(dst, sample, sample + sizeof(sample));

            dst.finish();
        }
        {
            test_utils::failed_recycle_call_tester<char> dst
                { {"abcdefghij", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 10} };

            strf::detail::output_buffer_interchar_copy(dst, sample, sample + sizeof(sample));

            dst.finish();
        }
        {
            test_utils::failed_recycle_call_tester<char> dst
                { {"abcdefghi", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 9} };

            strf::detail::output_buffer_interchar_copy(dst, sample, sample + sizeof(sample));

            dst.finish();
        }

        {
            test_utils::recycle_call_tester<char16_t> dst
                { {u"abcdefghij", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 5} };

            strf::detail::output_buffer_interchar_copy(dst, sample, sample + sizeof(sample));

            dst.finish();
        }
        {
            test_utils::failed_recycle_call_tester<char16_t> dst
                { {u"abcdefghij", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 10} };

            strf::detail::output_buffer_interchar_copy(dst, sample, sample + sizeof(sample));

            dst.finish();
        }
        {
            test_utils::failed_recycle_call_tester<char16_t> dst
                { {u"abcdefghi", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 9} };

            strf::detail::output_buffer_interchar_copy(dst, sample, sample + sizeof(sample));

            dst.finish();
        }

    }
    {   // Cover strf::detail::write_fill
        {
            test_utils::recycle_call_tester<char> dst
                { {"aaaaaaaaaa", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 5} };

            strf::detail::write_fill(dst, 10, 'a');
            TEST_TRUE(dst.good());
            dst.finish();
        }
        {
            test_utils::failed_recycle_call_tester<char> dst
                { {"aaaaaaaa", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 8} };

            strf::detail::write_fill(dst, 10, 'a');
            TEST_FALSE(dst.good());
            dst.finish();
        }
    }
    {
        // Cover write(output_buffer<CharT>&, const CharT*)
        test_utils::recycle_call_tester<char> dst
                { {"abcdefghij", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 5} };
        write(dst, "abcdefghij");
        dst.finish();
    }
    {
        // Cover put(output_buffer<CharT>&, CharT)
        test_utils::recycle_call_tester<char> dst
                { {"abcd", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 3} };

        // When put does not call recycle()
        strf::put(dst, 'a');
        strf::put(dst, 'b');
        strf::put(dst, 'c');

        // When it does
        strf::put(dst, 'd');

        dst.finish();
    }
    {
        // Cover output_buffer<CharT>::ensure()
        {
            test_utils::recycle_call_tester<char> dst
                { {"abcdef", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 4} };

            dst.ensure(2); // when argument is less than buffer_space()
            dst.buffer_ptr()[0] = 'a';
            dst.buffer_ptr()[1] = 'b';
            dst.advance(2);

            dst.ensure(2); // when argument is equal to buffer_space()
            dst.buffer_ptr()[0] = 'c';
            dst.buffer_ptr()[1] = 'd';
            dst.advance(2);

            dst.ensure(2);
            dst.buffer_ptr()[0] = 'e';
            dst.buffer_ptr()[1] = 'f';
            dst.advance(2);

            dst.finish();
        }
        {
            test_utils::recycle_call_tester<char> dst
                { {"abcd", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 2} };

            dst.ensure(4); // When argument is greater than buffer_space()
            dst.buffer_ptr()[0] = 'a';
            dst.buffer_ptr()[1] = 'b';
            dst.buffer_ptr()[2] = 'c';
            dst.buffer_ptr()[3] = 'd';
            dst.advance(4);

            dst.finish();
        }
    }
    {
        // When output_buffer<CharT>::do_write() calls recycle()
        // and the good() afterwards is false
        test_utils::failed_recycle_call_tester<char> dst
                { {"abcde", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 5} };

        dst.write("abcdefghij", 10);
        dst.finish();
    }
}
REGISTER_STRF_TEST(test_output_buffer_functions)

