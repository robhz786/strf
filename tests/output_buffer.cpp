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
            test_utils::recycle_call_tester<char> dest
                { {"abcdefghij", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 5} };

            strf::detail::output_buffer_interchar_copy(dest, sample, sizeof(sample));

            dest.finish();
        }
        {
            test_utils::failed_recycle_call_tester<char> dest
                { {"abcdefghij", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 10} };

            strf::detail::output_buffer_interchar_copy(dest, sample, sizeof(sample));

            dest.finish();
        }
        {
            test_utils::failed_recycle_call_tester<char> dest
                { {"abcdefghi", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 9} };

            strf::detail::output_buffer_interchar_copy(dest, sample, sizeof(sample));

            dest.finish();
        }

        {
            test_utils::recycle_call_tester<char16_t> dest
                { {u"abcdefghij", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 5} };

            strf::detail::output_buffer_interchar_copy(dest, sample, sizeof(sample));

            dest.finish();
        }
        {
            test_utils::failed_recycle_call_tester<char16_t> dest
                { {u"abcdefghij", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 10} };

            strf::detail::output_buffer_interchar_copy(dest, sample, sizeof(sample));

            dest.finish();
        }
        {
            test_utils::failed_recycle_call_tester<char16_t> dest
                { {u"abcdefghi", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 9} };

            strf::detail::output_buffer_interchar_copy(dest, sample, sizeof(sample));

            dest.finish();
        }

    }
    {   // Cover strf::detail::write_fill
        {
            test_utils::recycle_call_tester<char> dest
                { {"aaaaaaaaaa", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 5} };

            strf::detail::write_fill(dest, 10, 'a');
            TEST_TRUE(dest.good());
            dest.finish();
        }
        {
            test_utils::failed_recycle_call_tester<char> dest
                { {"aaaaaaaa", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 8} };

            strf::detail::write_fill(dest, 10, 'a');
            TEST_FALSE(dest.good());
            dest.finish();
        }
    }
    {
        // Cover write(output_buffer<CharT>&, const CharT*)
        test_utils::recycle_call_tester<char> dest
                { {"abcdefghij", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 5} };
        write(dest, "abcdefghij");
        dest.finish();
    }
    {
        // Cover put(output_buffer<CharT>&, CharT)
        test_utils::recycle_call_tester<char> dest
                { {"abcd", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 3} };

        // When put does not call recycle()
        strf::put(dest, 'a');
        strf::put(dest, 'b');
        strf::put(dest, 'c');

        // When it does
        strf::put(dest, 'd');

        dest.finish();
    }
    {
        // Cover output_buffer<CharT>::ensure()
        {
            test_utils::recycle_call_tester<char> dest
                { {"abcdef", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 4} };

            dest.ensure(2); // when argument is less than buffer_space()
            dest.buffer_ptr()[0] = 'a';
            dest.buffer_ptr()[1] = 'b';
            dest.advance(2);

            dest.ensure(2); // when argument is equal to buffer_space()
            dest.buffer_ptr()[0] = 'c';
            dest.buffer_ptr()[1] = 'd';
            dest.advance(2);

            dest.ensure(2);
            dest.buffer_ptr()[0] = 'e';
            dest.buffer_ptr()[1] = 'f';
            dest.advance(2);

            dest.finish();
        }
        {
            test_utils::recycle_call_tester<char> dest
                { {"abcd", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 2} };

            dest.ensure(4); // When argument is greater than buffer_space()
            dest.buffer_ptr()[0] = 'a';
            dest.buffer_ptr()[1] = 'b';
            dest.buffer_ptr()[2] = 'c';
            dest.buffer_ptr()[3] = 'd';
            dest.advance(4);

            dest.finish();
        }
    }
    {
        // When output_buffer<CharT>::do_write() calls recycle()
        // and the good() afterwards is false
        test_utils::failed_recycle_call_tester<char> dest
                { {"abcde", BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, 5} };

        dest.write("abcdefghij", 10);
        dest.finish();
    }
}
REGISTER_STRF_TEST(test_output_buffer_functions)

